mutable struct SplitIntegrationCallback{A, K}
    alg            :: A
    stage_coupling :: Bool
    kwargs         :: K
end

@doc raw"""
    SplitIntegrationCallback(alg; stage_coupling=false, kwargs...)

Callback to integrate the `TotalLagrangianSPHSystem`s in a `Semidiscretization`
separately from the other systems.
After each time step of the main integrator (in which TLSPH systems are ignored),
the TLSPH systems are integrated for multiple smaller time steps with their own integrator.

This is useful if the TLSPH systems require much smaller time steps than the fluid systems,
which is usually the case when stiff materials are simulated.
It is especially useful if additionally the number of TLSPH particles is much smaller
than the number of fluid particles, so that a fluid time step is much more expensive
than a TLSPH substep.

For fluid-structure interactions with stiff materials like metal or carbon fiber
composites, this can lead to significant speedups of several hundred times if the ratio
of fluid to solid particles is large enough (e.g. 100:1 or more).

# Arguments
- `alg`: The time integration algorithm to use for the TLSPH systems.

# Keywords
- `stage_coupling=false`: If `false`, the TLSPH systems are only updated between full
                          time steps of the main integrator.
                          If `true`, the TLSPH systems are integrated to the intermediate
                          stage times of the main integrator as well. The sub-integration
                          starts from the solution at last full time step for each stage.
                          This is significantly more expensive, but restores the stability
                          properties of the main time integrator.
                          For large time step size ratios, `stage_coupling=false` might
                          require a significantly smaller time step size for stability
                          at the FSI interface.
- `kwargs...`: Additional keyword arguments passed to the integrator of the TLSPH systems.

# Examples
```jldoctest; output=false
using OrdinaryDiffEq

# Low-storage RK method with fixed step size
callback = SplitIntegrationCallback(CarpenterKennedy2N54(williamson_condition=false),
                                    dt=1e-5)

# RK method with automatic error-based step size control
callback = SplitIntegrationCallback(RDPK3SpFSAL49(), abstol=1e-6, reltol=1e-4)

# output
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ SplitIntegrationCallback                                                                         │
│ ════════════════════════                                                                         │
│ alg: …………………………………………………………………… RDPK3SpFSAL49                                                    │
│ abstol: …………………………………………………………… 1.0e-6                                                           │
│ reltol: …………………………………………………………… 0.0001                                                           │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```
"""
function SplitIntegrationCallback(alg; stage_coupling=false, kwargs...)
    split_integration_callback = SplitIntegrationCallback(alg, stage_coupling, kwargs)

    # The first one is the `condition`, the second the `affect!`
    return DiscreteCallback(split_integration_callback, split_integration_callback,
                            initialize=(initialize_split_integration!),
                            save_positions=(false, false))
end

function initialize_split_integration!(cb, vu_ode, t, integrator)
    semi = integrator.p.semi
    split_integration_callback = cb.affect!
    (; alg, stage_coupling, kwargs) = split_integration_callback

    # Disable TLSPH integration in the original integrator
    semi.integrate_tlsph[] = false

    # Create split integrator with TLSPH systems only
    systems = filter(i -> i isa TotalLagrangianSPHSystem, semi.systems)

    if isempty(systems)
        throw(ArgumentError("`SplitIntegrationCallback` must be used with a " *
                            "`TotalLagrangianSPHSystem`"))
    end

    # These neighborhood searches are never used
    periodic_box = extract_periodic_box(semi.neighborhood_searches[1][1])
    neighborhood_search = TrivialNeighborhoodSearch{ndims(first(systems))}(; periodic_box)
    semi_split = Semidiscretization(systems...,
                                    neighborhood_search=neighborhood_search,
                                    parallelization_backend=semi.parallelization_backend)

    sizes_u = (u_nvariables(system) * n_integrated_particles(system) for system in systems)
    sizes_v = (v_nvariables(system) * n_integrated_particles(system) for system in systems)

    v_ode, u_ode = vu_ode.x
    v0_ode_split = similar(v_ode, sum(sizes_v))
    u0_ode_split = similar(u_ode, sum(sizes_u))

    # A buffer to store interaction forces between structure and fluid
    dv_ode_split = similar(v0_ode_split)

    # Copy the initial conditions to the split integrator
    copy_to_split!(v0_ode_split, u0_ode_split, semi_split, v_ode, u_ode, semi)

    # A zero `tspan` sets `tdir` to zero, which breaks adding tstops.
    # Therefore, we have to use an arbitrary non-zero `tspan` here,
    # but we remove the final tstop later.
    tspan = (t, t + 1)

    # Payload that we need inside the split `kick!` function
    p = (; semi=semi_split, semi_large=semi, dv_ode_split)
    ode_split = DynamicalODEProblem(kick_split!, drift_split!, v0_ode_split, u0_ode_split,
                                    tspan, p)

    # Add lightweight callback to (potentially) update the averaged velocity
    # during the split integration.
    callback = UpdateAveragedVelocityCallback()
    if haskey(kwargs, :callback)
        kwargs[:callback] = CallbackSet(kwargs[:callback], callback)
    else
        kwargs[:callback] = callback
    end

    # Create the split integrator.
    # We need the timer here to keep the output clean because this will call `kick!` once.
    @trixi_timeit timer() "split integration" begin
        @trixi_timeit timer() "init" begin
            TimerOutputs.@notimeit timer() begin
                split_integrator = SciMLBase.init(ode_split, alg; save_everystep=false,
                                                  kwargs...)
            end

            # Remove the `tstop` for the final time (equivalent to zero `tspan`)
            SciMLBase.pop_tstop!(split_integrator)
        end
    end

    # Store the required data as payload in `integrator.p`.
    # `integrator.p` is a `NamedTuple` that contains `p.semi`
    # and `p.split_integration_data`. The latter is usually `nothing`, but we can use it
    # to store another `NamedTuple` containing all split integration data.
    vu_ode_split = copy(split_integrator.u)
    payload = (; stage_coupling, integrator=split_integrator, vu_ode_split, t_ref=Ref(t))
    integrator.p = (; semi, split_integration_data=payload)

    return cb
end

# `condition`
function (split_integration_callback::SplitIntegrationCallback)(u, t, integrator)
    # Integrate TLSPH after every time step
    return true
end

# `affect!`
function (split_integration_callback::SplitIntegrationCallback)(integrator)
    new_t = integrator.t
    v_ode, u_ode = integrator.u.x

    @trixi_timeit timer() "split integration" begin
        # Update quantities that are stored in the systems. These quantities (e.g. pressure)
        # still have the values from the last stage of the previous step if not updated here.
        @trixi_timeit timer() "update systems and nhs" begin
            update_systems_and_nhs(v_ode, u_ode, integrator.p.semi, new_t)
        end
    end

    # Advance the split integrator to the new time
    split_integrate!(v_ode, u_ode, new_t, integrator.p.split_integration_data)

    # Update split solution stored in `integrator.p`
    data = integrator.p.split_integration_data
    data.vu_ode_split .= data.integrator.u
    data.t_ref[] = new_t

    # Tell OrdinaryDiffEq that `u` has NOT been modified.
    # Theoretically, the TLSPH part has been modified, but since TLSPH is integrated
    # separately, this part is never used by the main integrator (dv = du = 0).
    # With this trick, we can avoid unnecessarily re-computing FSAL stages.
    u_modified!(integrator, false)
end

# No `SplitIntegrationCallback` used
function split_integrate_stage!(v_ode, u_ode, t, split_integration_data::Nothing)
    return v_ode
end

function split_integrate_stage!(v_ode, u_ode, t, split_integration_data)
    (; stage_coupling) = split_integration_data

    if stage_coupling
        split_integrate!(v_ode, u_ode, t, split_integration_data)
    end

    return v_ode
end

function split_integrate!(v_ode, u_ode, new_t, split_integration_data)
    old_t = split_integration_data.t_ref[]
    if new_t <= old_t
        # First stage is usually called at the same time as the last time step.
        # Nothing to do here.
        return v_ode
    end

    vu_ode_split = split_integration_data.vu_ode_split
    split_integrator = split_integration_data.integrator
    semi_split = split_integrator.p.semi
    dv_ode_split = split_integrator.p.dv_ode_split
    semi_large = split_integrator.p.semi_large

    @trixi_timeit timer() "split integration" begin
        @trixi_timeit timer() "init" begin
            TimerOutputs.@notimeit timer() begin
                SciMLBase.reinit!(split_integrator, vu_ode_split; t0=old_t, tf=new_t)
            end
        end

        # Compute structure-fluid interaction forces
        @trixi_timeit timer() "system interaction" begin
            set_zero!(dv_ode_split)

            other_interaction_split!(dv_ode_split, semi_large, v_ode, u_ode, semi_split)
        end

        # Integrate the split integrator up to the new time
        SciMLBase.solve!(split_integrator)

        v_ode_split, u_ode_split = split_integrator.u.x

        # Copy the solutions back to the original integrator
        @trixi_timeit timer() "copy back" copy_from_split!(v_ode, u_ode,
                                                           v_ode_split, u_ode_split,
                                                           semi_large, semi_split)
    end

    return v_ode
end

function kick_split!(dv_ode_split, v_ode_split, u_ode_split, p, t)
    semi_large = p.semi_large
    semi_split = p.semi

    @trixi_timeit timer() "reset ∂v/∂t" set_zero!(dv_ode_split)

    # Update the TLSPH systems
    @trixi_timeit timer() "update systems and nhs" begin
        update_systems_split!(semi_split, v_ode_split, u_ode_split, t)
    end

    # Only compute structure-structure self-interaction.
    # structure-fluid interaction forces are computed once before the split time integration
    # loop and are applied below.
    @trixi_timeit timer() "system interaction" begin
        self_interaction_split!(dv_ode_split, v_ode_split, u_ode_split,
                                semi_split, semi_large)
    end

    # Add structure-fluid interaction forces
    dv_ode_split .+= p.dv_ode_split

    @trixi_timeit timer() "source terms" add_source_terms!(dv_ode_split, v_ode_split,
                                                           u_ode_split, semi_large, t;
                                                           semi_wrap=semi_split)
end

function drift_split!(du_ode, v_ode, u_ode, p, t)
    @trixi_timeit timer() "drift!" begin
        # Avoid cluttering the timer output with sub-timers of `drift!`
        TimerOutputs.@notimeit timer() begin
            drift!(du_ode, v_ode, u_ode, p, t)
        end
    end
end

# Update the systems before calling `interact!` to compute forces
function update_systems_split!(semi_split, v_ode_split, u_ode_split, t)
    # First update step before updating the NHS.
    # This is used for writing the current coordinates into the TLSPH system.
    foreach_system(semi_split) do system
        v = wrap_v(v_ode_split, system, semi_split)
        u = wrap_u(u_ode_split, system, semi_split)

        update_positions!(system, v, u, v_ode_split, u_ode_split, semi_split, t)
    end

    # Second update step.
    # This is used to calculate the deformation gradient and stress tensor.
    foreach_system(semi_split) do system
        v = wrap_v(v_ode_split, system, semi_split)
        u = wrap_u(u_ode_split, system, semi_split)

        update_quantities!(system, v, u, v_ode_split, u_ode_split, semi_split, t)
    end

    # The `TotalLagrangianSPHSystem` doesn't have an `update_pressure!` method

    # No `update_boundary_interpolation!` for performance reasons, or we will lose
    # a lot of the speedup that we can gain with split integration.
    # We assume that the TLSPH particles move so little during the substeps
    # that the extrapolated pressure/density values can be treated as constant.

    # The `TotalLagrangianSPHSystem` doesn't have an `update_final!` method
end

function self_interaction_split!(dv_ode_split, v_ode_split, u_ode_split, semi_split, semi)
    # Only loop over (TLSPH) systems in the split integrator
    foreach_system(semi_split) do system
        # Construct string for the interactions timer.
        # Avoid allocations from string construction when no timers are used.
        # TODO do we need to disable timers in split integration?
        if timeit_debug_enabled()
            system_index = system_indices(system, semi)
            timer_str = "$(timer_name(system))$system_index-$(timer_name(system))$system_index"
        else
            timer_str = ""
        end

        dv = wrap_v(dv_ode_split, system, semi_split)
        v = wrap_v(v_ode_split, system, semi_split)
        u = wrap_u(u_ode_split, system, semi_split)

        @trixi_timeit timer() timer_str begin
            interact!(dv, v, u, v, u, system, system, semi; integrate_tlsph=true)
        end
    end
end

function other_interaction_split!(dv_ode_split, semi, v_ode, u_ode, semi_split)
    # Only loop over (TLSPH) systems in the split integrator
    foreach_system(semi_split) do system
        # Loop over all neighbors in the big integrator
        foreach_system(semi) do neighbor
            if system === neighbor
                # Only compute interaction with other systems
                return
            end

            # Construct string for the interactions timer.
            # Avoid allocations from string construction when no timers are used.
            if timeit_debug_enabled()
                system_index = system_indices(system, semi)
                neighbor_index = system_indices(neighbor, semi)
                timer_str = "$(timer_name(system))$system_index-$(timer_name(neighbor))$neighbor_index"
            else
                timer_str = ""
            end

            dv = wrap_v(dv_ode_split, system, semi_split)
            v_system = wrap_v(v_ode, system, semi)
            u_system = wrap_u(u_ode, system, semi)

            v_neighbor = wrap_v(v_ode, neighbor, semi)
            u_neighbor = wrap_u(u_ode, neighbor, semi)

            @trixi_timeit timer() timer_str begin
                interact!(dv, v_system, u_system, v_neighbor, u_neighbor,
                          system, neighbor, semi; integrate_tlsph=true)
            end
        end
    end

    return dv_ode_split
end

# Copy the solution from the large integrator to the split integrator
@inline function copy_to_split!(v_ode_split, u_ode_split, semi_split, v_ode, u_ode, semi)
    foreach_system(semi_split) do system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)
        v_split = wrap_v(v_ode_split, system, semi_split)
        u_split = wrap_u(u_ode_split, system, semi_split)

        @threaded semi for particle in each_integrated_particle(system)
            for i in axes(v, 1)
                v_split[i, particle] = v[i, particle]
            end

            for i in axes(u, 1)
                u_split[i, particle] = u[i, particle]
            end
        end
    end
end

# Copy the solution from the split integrator to the large integrator
@inline function copy_from_split!(v_ode, u_ode, v_ode_split, u_ode_split, semi, semi_split)
    foreach_system(semi_split) do system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)
        v_split = wrap_v(v_ode_split, system, semi_split)
        u_split = wrap_u(u_ode_split, system, semi_split)

        @threaded semi for particle in each_integrated_particle(system)
            for i in axes(v, 1)
                v[i, particle] = v_split[i, particle]
            end

            for i in axes(u, 1)
                u[i, particle] = u_split[i, particle]
            end
        end
    end
end

function calculate_dt(v_ode, u_ode, cfl_number, p::NamedTuple, integrate_tlsph)
    # The split integrator contains a `NamedTuple`
    return calculate_dt(v_ode, u_ode, cfl_number, p.semi_split, integrate_tlsph)
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:SplitIntegrationCallback})
    @nospecialize cb # reduce precompilation time
    print(io, "SplitIntegrationCallback(alg=", cb.affect!.alg,
          ", stage_coupling=", cb.affect!.stage_coupling, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:SplitIntegrationCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        split_cb = cb.affect!
        setup = [
            "alg" => split_cb.alg |> typeof |> nameof |> string,
            "stage_coupling" => split_cb.stage_coupling
        ]

        for (key, value) in split_cb.kwargs
            push!(setup, string(key) => string(value))
        end

        summary_box(io, "SplitIntegrationCallback", setup)
    end
end

# === Non-public callback for updating the averaged velocity ===
# When no split integration is used, this is done from the `UpdateCallback`.
# With split integration, we use this lightweight callback to avoid updating the systems.
function UpdateAveragedVelocityCallback()
    # The first one is the `condition`, the second the `affect!`
    return DiscreteCallback(update_averaged_velocity_callback!,
                            update_averaged_velocity_callback!,
                            initialize=(initialize_averaged_velocity_callback!),
                            save_positions=(false, false))
end

# `initialize`
function initialize_averaged_velocity_callback!(cb, vu_ode, t, integrator)
    v_ode, u_ode = vu_ode.x
    semi = integrator.p

    foreach_system(semi) do system
        initialize_averaged_velocity!(system, v_ode, semi, t)
    end

    return cb
end

# `condition`
function update_averaged_velocity_callback!(u, t, integrator)
    return true
end

# `affect!`
function update_averaged_velocity_callback!(integrator)
    t_new = integrator.t
    semi = integrator.p
    v_ode, u_ode = integrator.u.x

    foreach_system(semi) do system
        compute_averaged_velocity!(system, v_ode, semi, t_new)
    end

    # Tell OrdinaryDiffEq that `integrator.u` has not been modified
    u_modified!(integrator, false)

    return integrator
end

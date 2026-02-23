mutable struct SplitIntegrationCallback{A, K}
    alg               :: A
    stage_coupling    :: Bool
    predict_positions :: Bool
    kwargs            :: K
end

@doc raw"""
    SplitIntegrationCallback(alg; stage_coupling=false, predict_positions=true, kwargs...)

Callback to integrate the `TotalLagrangianSPHSystem`s in a `Semidiscretization`
separately from the other systems.
For each time step of the main integrator (in which TLSPH systems are ignored),
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
- `stage_coupling=false`:   If `false`, the TLSPH systems are only updated between full
                            time steps of the main integrator.
                            If `true`, the TLSPH systems are integrated to the intermediate
                            stage times of the main integrator as well. The sub-integrator
                            integrates from the previous fluid stage time to the next stage
                            time, using the intermediate stage predictions for the fluid
                            state. This strategy is highly efficient (no sub-steps have to be
                            repeated) but less accurate than repeating the sub-integration
                            with the final (as opposed to predicted) fluid state.
                            Note that this type of stage-level coupling is still more accurate
                            than step-level coupling (`stage_coupling=false`).
                            For large time step size ratios, `stage_coupling=false` might
                            require a significantly (often 2x) smaller fluid time step size
                            for stability at the FSI interface.
                            For small time step size ratios, `stage_coupling=false` might be
                            sufficiently stable and more efficient than `stage_coupling=true`.
                            Note that `stage_coupling=true` is only compatible with fluid
                            time integration schemes that have monotonically increasing
                            stage times and no stage time smaller than the time
                            of the previous full step.
- `predict_positions=true`: The force on the structure due to the fluid is kept constant
                            during one sub-integration call. When computing this force,
                            the new fluid state and the old structure state are available.
                            To avoid inconsistencies and improve accuracy (not stability),
                            we can predict the structure positions at the new time with
                            a simple Euler step,
                            ``u \leftarrow u + v\,(t_{\mathrm{new}} - t_{\mathrm{previous}})``,
                            only for this fluid force calculation.
                            If `false`, use the old structure state together with the new
                            fluid state. If `true`, predict the structure positions for the
                            fluid force calculation.
- `kwargs...`:              Additional keyword arguments passed to the integrator
                            of the TLSPH systems. Use this for callbacks like the
                            [`StepsizeCallback`](@ref) for choosing the sub-integration
                            time step.


# Examples
```jldoctest; output=false
using OrdinaryDiffEq

# Low-storage RK method with CFL condition for time step size
callback = SplitIntegrationCallback(CarpenterKennedy2N54(williamson_condition=false),
                                    dt=1.0, # This is overwritten by the stepsize callback
                                    callback=StepsizeCallback(cfl=1.6),
                                    stage_coupling=true)

# output
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ SplitIntegrationCallback                                                                         │
│ ════════════════════════                                                                         │
│ alg: …………………………………………………………………… CarpenterKennedy2N54                                             │
│ stage_coupling: ……………………………………… true                                                             │
│ predict_positions: ……………………………… true                                                             │
│ dt: ……………………………………………………………………… 1.0                                                              │
│ callback: ……………………………………………………… StepsizeCallback(is_constant=true, cfl_number=1.6)               │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```
"""
function SplitIntegrationCallback(alg; stage_coupling=false, predict_positions=true,
                                  kwargs...)
    # Add lightweight callback to (potentially) update the averaged velocity
    # during the split integration.
    if haskey(kwargs, :callback)
        # Note that `CallbackSet`s can be nested
        kwargs = (; kwargs..., callback=CallbackSet(values(kwargs).callback,
                                                    UpdateAveragedVelocityCallback()))
    else
        kwargs = (; kwargs..., callback=UpdateAveragedVelocityCallback())
    end
    split_integration_callback = SplitIntegrationCallback(alg, stage_coupling,
                                                          predict_positions, pairs(kwargs))

    # The first one is the `condition`, the second the `affect!`
    return DiscreteCallback(split_integration_callback, split_integration_callback,
                            initialize=(initialize_split_integration!),
                            save_positions=(false, false))
end

function initialize_split_integration!(cb, vu_ode, t, integrator)
    semi = integrator.p.semi
    split_integration_callback = cb.affect!
    (; alg, stage_coupling, predict_positions, kwargs) = split_integration_callback

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

    # Create the split integrator.
    # We need the timer here to keep the output clean because this will call `kick!` once.
    @trixi_timeit timer() "split integration" begin
        @trixi_timeit timer() "init" begin
            TimerOutputs.@notimeit timer() begin
                # Use `save_everystep=false` to avoid saving multiple copies
                # of `v_ode` and `u_ode`.
                # We set the final time and solve the integrator to that time in each split
                # integration call, which will save the final solution if we don't set
                # `save_end=false`, leading to one copy of the split state per fluid stage.
                split_integrator = SciMLBase.init(ode_split, alg; save_everystep=false,
                                                  save_end=false, kwargs...)
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
    payload = (; stage_coupling, predict_positions, integrator=split_integrator,
               large_integrator=integrator, vu_ode_split, t_ref=Ref(t), n_reject=Ref(0))
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
    data = integrator.p.split_integration_data
    data.n_reject[] = integrator.stats.nreject

    # Advance the split integrator to the new time
    split_integrate!(v_ode, u_ode, new_t, data)

    # Update split solution stored in `integrator.p`
    data.vu_ode_split .= data.integrator.u
    data.t_ref[] = new_t

    if data.stage_coupling
        # Tell OrdinaryDiffEq that `u` has NOT been modified.
        # Theoretically, the TLSPH part has been modified, but in the FSAL case,
        # the time at the last stage is the same as the step time, so the split integration
        # above is skipped and `u` is not modified at all.
        # Therefore, the derivative at the last stage can be reused for the next step.
        # TODO is `u_modified` ever relevant for non-FSAL methods?
        u_modified!(integrator, false)
    else
        # Tell OrdinaryDiffEq that `u` has been modified
        u_modified!(integrator, true)
    end

    return integrator
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

function split_integrate!(v_ode, u_ode, t_new, data)
    split_integrator = data.integrator

    t_previous = split_integrator.t

    restart = false
    rejected = data.large_integrator.stats.nreject > data.n_reject[]
    data.n_reject[] = data.large_integrator.stats.nreject
    if rejected
        # The previous time step was rejected.
        # We have to restart the split integration from the last full time step.
        # Note that we even have to do this if `t_new == t_previous` because
        # the split integrator was advanced to `t_new` with a rejected fluid state.
        restart = true

        # Make sure we don't have to go back before the last time step
        @assert t_new >= data.t_ref[]
    elseif t_new < t_previous - eps(t_previous)
        # @info "" t_new - t_previous data.large_integrator.dt t_new - data.t_ref[]
        # The stage time is smaller than the previous stage/step time, but the last step
        # was not rejected. This means that either the RK scheme contains a negative
        # node value (requesting a stage time before the time of the last full step)
        # or the node values are non-monotonic (requesting a stage time before the previous
        # stage time). In both cases, the scheme is not suitable for stage-level coupling
        # in the form implemented here.
        msg = "stage-level coupling with `SplitIntegrationCallback` requires that stage " *
              "times are monotonically increasing and that no stage time is smaller " *
              "than the time of the last full step. This time integration scheme can " *
              "only be used with `stage_coupling=false` when creating the " *
              "`SplitIntegrationCallback`. It is recommended to use a different time " *
              "integration scheme with monotonic stage times."
        throw(ArgumentError(msg))
    end

    if restart
        # Restart the split integration from the last full time step
        t_previous = data.t_ref[]
        vu_ode_split = data.vu_ode_split
    else
        # Continue the split integration from the previous stage
        vu_ode_split = split_integrator.u
    end

    semi_split = split_integrator.p.semi
    dv_ode_split = split_integrator.p.dv_ode_split
    semi_large = split_integrator.p.semi_large

    @trixi_timeit timer() "split integration" begin
        # Copy the solutions back to the original integrator.
        # We modify `v_ode` and `u_ode`, which is technically not allowed during stages,
        # hence there are no guarantees about the structure part of `v_ode` and `u_ode`.
        # By copying the current split integration values, we make sure that it's correct.
        predict_positions = Val(data.predict_positions)
        @trixi_timeit timer() "copy back" copy_from_split!(v_ode, u_ode,
                                                           vu_ode_split.x...,
                                                           semi_large, semi_split,
                                                           t_new, t_previous;
                                                           predict_positions)
    end

    if !rejected && isapprox(t_new, t_previous)
        # This stage time is the same as the previous stage time and we don't need to
        # re-initialize the split integrator due to a rejected step.
        # There is nothing to do here.
        # IMPORTANT: This has to be after copying the solution to the large integrator.
        # Otherwise, the large integrator might contain arbitrary values since we
        # are modifying `v_ode` and `u_ode` during stages, which is not allowed.
        # IMPORTANT: If we try to `return` from within the `@trixi_timeit` block,
        # the timer output will be messed up for some reason.
        return true
    end

    @trixi_timeit timer() "split integration" begin
        if restart
            # Reset the split integrator to the state at the last full time step
            @trixi_timeit timer() "init" begin
                TimerOutputs.@notimeit timer() begin
                    SciMLBase.reinit!(split_integrator, vu_ode_split;
                                      t0=t_previous, tf=t_new)
                end
            end
        else
            # Continue from the previous state
            add_tstop!(split_integrator, t_new)
        end

        # Update the large semidiscretization with the predicted structure positions
        @trixi_timeit timer() "update systems and nhs" begin
            update_systems_and_nhs(v_ode, u_ode, semi_large, t_new)
        end

        # Compute structure-fluid interaction forces that are kept constant
        # when applied during the split integration.
        @trixi_timeit timer() "system interaction" begin
            set_zero!(dv_ode_split)

            other_interaction_split!(dv_ode_split, semi_large, v_ode, u_ode, semi_split)
        end

        # Integrate the split integrator up to the new time
        sol = SciMLBase.solve!(split_integrator)
        if sol.retcode != SciMLBase.ReturnCode.Success
            error("`SplitIntegrationCallback` failed with return code $(sol.retcode). " *
                  "Try reducing the split integration time step size.")
        end

        # Copy the solutions back to the original integrator
        @trixi_timeit timer() "copy back" copy_from_split!(v_ode, u_ode,
                                                           split_integrator.u.x...,
                                                           semi_large, semi_split,
                                                           t_new, t_previous)
    end

    return true
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
    # structure-fluid interaction forces are computed once
    # before the split time integration loop and are applied below.
    @trixi_timeit timer() "system interaction" begin
        self_interaction_split!(dv_ode_split, v_ode_split, u_ode_split,
                                semi_split, semi_large)
    end

    # Add structure-fluid interaction forces
    dv_ode_split .+= p.dv_ode_split

    add_source_terms!(dv_ode_split, v_ode_split, u_ode_split, semi_large, t;
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
@inline function copy_from_split!(v_ode, u_ode, v_ode_split, u_ode_split, semi, semi_split,
                                  t_new, t_previous;
                                  predict_positions::Val{PREDICT}=Val(false)) where {PREDICT}
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
                if PREDICT
                    # Predict positions at `t_new` with a simple Euler step
                    u[i, particle] += v[i, particle] * (t_new - t_previous)
                end
            end
        end
    end
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:SplitIntegrationCallback})
    @nospecialize cb # reduce precompilation time
    print(io, "SplitIntegrationCallback(alg=", cb.affect!.alg,
          ", stage_coupling=", cb.affect!.stage_coupling,
          ", predict_positions=", cb.affect!.predict_positions, ")")
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
            "stage_coupling" => split_cb.stage_coupling,
            "predict_positions" => split_cb.predict_positions
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

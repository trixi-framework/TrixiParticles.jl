mutable struct SplitIntegrationCallback
    integrator :: Any
    alg        :: Any
    kwargs     :: Any
end

@doc raw"""
    SplitIntegrationCallback()

Callback to...
"""
function SplitIntegrationCallback(alg; kwargs...)
    split_integration_callback = SplitIntegrationCallback(nothing, alg, kwargs)

    # The first one is the `condition`, the second the `affect!`
    return DiscreteCallback(split_integration_callback, split_integration_callback,
                            initialize=(initialize_split_integration!),
                            save_positions=(false, false))
end

function initialize_split_integration!(cb, u, t, integrator)
    semi = integrator.p
    split_integration_callback = cb.affect!
    (; alg, kwargs) = split_integration_callback

    # Disable TLSPH integration in the original integrator
    semi.integrate_tlsph[] = false

    # Create split integrator with TLSPH systems only
    systems = filter(i -> i isa TotalLagrangianSPHSystem, semi.systems)

    # These neighborhood searches are never used
    semi_split = Semidiscretization(systems...,
                                    neighborhood_search=TrivialNeighborhoodSearch{ndims(first(systems))}(),
                                    parallelization_backend=semi.parallelization_backend)

    sizes_u = (u_nvariables(system) * n_moving_particles(system) for system in systems)
    sizes_v = (v_nvariables(system) * n_moving_particles(system) for system in systems)

    v_ode, u_ode = integrator.u.x
    v0_ode_split = similar(v_ode, sum(sizes_v))
    u0_ode_split = similar(u_ode, sum(sizes_u))

    # Copy the initial conditions to the split integrator
    copy_to_split!(v_ode, u_ode, v0_ode_split, u0_ode_split, semi, semi_split)

    # A zero `tspan` sets `tdir` to zero, which breaks adding tstops
    tspan = (integrator.t, integrator.t + 1)
    p = (; v_ode, u_ode, semi, semi_split)
    ode_split = DynamicalODEProblem(kick_split!, drift_split!, v0_ode_split, u0_ode_split,
                                    tspan, p)

    # Create the split integrator.
    # We need the timer here to keep the output clean because this will call `kick!` once.
    @trixi_timeit timer() "split integration" begin
        @trixi_timeit timer() "init" begin
            TimerOutputs.@notimeit timer() split_integrator = SciMLBase.init(ode_split, alg; save_everystep=false, kwargs...)
        end
    end

    # Remove the `tstop` (equivalent to zero `tspan`)
    SciMLBase.pop_tstop!(split_integrator)

    # Store the integrator in the callback
    split_integration_callback.integrator = split_integrator

    return cb
end

# `condition`
function (split_integration_callback::SplitIntegrationCallback)(u, t, integrator)
    # Integrate TLSPH after every time step
    return true
end

# `affect!`
function (split_integration_callback::SplitIntegrationCallback)(integrator)
    # Function barrier for type stability
    affect_inner!(integrator, split_integration_callback.integrator)
end

function affect_inner!(integrator, split_integrator)
    semi_split = split_integrator.p.semi_split

    semi = integrator.p
    new_t = integrator.t

    v_ode, u_ode = integrator.u.x

    @assert semi == split_integrator.p.semi
    split_integrator.p = (; v_ode, u_ode, semi, semi_split)

    @trixi_timeit timer() "split integration" begin
        # Update quantities that are stored in the systems. These quantities (e.g. pressure)
        # still have the values from the last stage of the previous step if not updated here.
        @trixi_timeit timer() "update systems and nhs" update_systems_and_nhs(v_ode, u_ode,
                                                                              semi, new_t;
                                                                              update_from_callback=true)

        # Integrate the split integrator to the new time
        add_tstop!(split_integrator, new_t)
        @trixi_timeit timer() "solve" SciMLBase.solve!(split_integrator)

        v_ode_split, u_ode_split = split_integrator.u.x

        # Copy the solutions back to the original integrator
        @trixi_timeit timer() "copy back" copy_from_split!(v_ode, u_ode, v_ode_split, u_ode_split, semi, semi_split)
    end

    # Tell OrdinaryDiffEq that `u` has been modified
    u_modified!(integrator, true)

    return integrator
end

function kick_split!(dv_ode_split, v_ode_split, u_ode_split, p, t)
    (; v_ode, u_ode, semi, semi_split) = p

    @trixi_timeit timer() "reset ∂v/∂t" set_zero!(dv_ode_split)

    # Only loop over TLSPH systems
    @trixi_timeit timer() "copy to large v,u" copy_from_split!(v_ode, u_ode,
                                                               v_ode_split, u_ode_split,
                                                               semi, semi_split)

    # Update the TLSPH systems with the other systems as neighbors
    update_nhs_fun = (semi,
                      u_ode) -> update_nhs_split!(semi, u_ode, u_ode_split,
                                                  semi_split)
    @trixi_timeit timer() "update systems and nhs" update_systems_and_nhs(v_ode, u_ode,
                                                                          semi, t;
                                                                          systems=semi_split.systems,
                                                                          update_nhs_fun)

    @trixi_timeit timer() "system interaction" begin
        # Only loop over systems in the split integrator
        foreach_system(semi_split) do system
            # Loop over all neighbors in the big integrator
            foreach_system(semi) do neighbor
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
                v_system = wrap_v(v_ode_split, system, semi_split)
                u_system = wrap_u(u_ode_split, system, semi_split)

                v_neighbor = wrap_v(v_ode, neighbor, semi)
                u_neighbor = wrap_u(u_ode, neighbor, semi)

                @trixi_timeit timer() timer_str begin
                    interact!(dv, v_system, u_system, v_neighbor, u_neighbor,
                              system, neighbor, semi; integrate_tlsph=true)
                end
            end
        end
    end

    @trixi_timeit timer() "source terms" add_source_terms!(dv_ode_split, v_ode_split,
                                                           u_ode_split, semi, t;
                                                           semi_wrap=semi_split)
end

function drift_split!(du_ode, v_ode, u_ode, p, t)
    drift!(du_ode, v_ode, u_ode, p.semi_split, t)
end

function update_nhs_split!(semi, u_ode, u_ode_split, semi_split)
    # Only loop over systems in the split integrator
    foreach_system(semi_split) do system
        u_system = wrap_u(u_ode_split, system, semi_split)

        # Loop over neighbors in the big integrator
        foreach_system(semi) do neighbor
            # Static NHS for solid-solid (same system) and no interaction for two distinct
            # solid systems (TODO).
            if !(neighbor isa TotalLagrangianSPHSystem)
                u_neighbor = wrap_u(u_ode, neighbor, semi)
                neighborhood_search = get_neighborhood_search(system, neighbor, semi)

                # Only the TLSPH particles are moving. All other systems are frozen.
                # Note that this does nothing when a grid NHS is used.
                update_nhs!(neighborhood_search, system, neighbor, u_system, u_neighbor,
                            semi; points_moving=(true, false))
            end
        end
    end
end

# Copy the solution from the large integrator to the split integrator
@inline function copy_to_split!(v_ode, u_ode, v_ode_split, u_ode_split, semi, semi_split)
    foreach_system(semi_split) do system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)
        v_split = wrap_v(v_ode_split, system, semi_split)
        u_split = wrap_u(u_ode_split, system, semi_split)

        @threaded semi for particle in each_moving_particle(system)
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

        @threaded semi for particle in each_moving_particle(system)
            for i in axes(v, 1)
                v[i, particle] = v_split[i, particle]
            end

            for i in axes(u, 1)
                u[i, particle] = u_split[i, particle]
            end
        end
    end
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:SplitIntegrationCallback})
    @nospecialize cb # reduce precompilation time
    print(io, "SplitIntegrationCallback(alg=", cb.affect!.alg, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:SplitIntegrationCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        update_cb = cb.affect!
        setup = [
            "alg" => update_cb.alg
        ]
        summary_box(io, "SplitIntegrationCallback", setup)
    end
end

# Wrapper for any neighborhood search handler that forwards `get_neighborhood_search`
# to the wrapped handler, but doesn't do anything in the update step.
# This is used in the example tests to test for zero allocations in the `kick!` function.
struct NoUpdateNHSHandler{H} <: TrixiParticles.AbstractNHSHandler
    handler::H
end

@inline function TrixiParticles.get_neighborhood_search(handler::NoUpdateNHSHandler,
                                                        system_index, neighbor_index,
                                                        search_radius)
    return TrixiParticles.get_neighborhood_search(handler.handler, system_index,
                                                  neighbor_index, search_radius)
end

# No update.
function TrixiParticles.update_nhs!(semi, handler::NoUpdateNHSHandler, u_ode)
    return semi
end

# Copy a `Semidiscretization`, but wrap the neighborhood search handler with
# `NoUpdateNHSHandler`.
function copy_semi_with_no_update_nhs(semi)
    neighborhood_search_handler = NoUpdateNHSHandler(semi.neighborhood_search_handler)

    # Also make sure to use a `SerialBackend` to avoid allocations from multithreading.
    return Semidiscretization(semi.systems, semi.ranges_u, semi.ranges_v,
                              neighborhood_search_handler, semi.interaction_matrix,
                              SerialBackend(), Ref(true), Ref(true))
end

mutable struct MockIntegrator
    p::Any
    stats::Any
end

# Count allocations of one call to the right-hand side (`kick!` + `drift!`)
function count_rhs_allocations(sol; split_integration=nothing)
    t = sol.t[end]
    v_ode_, u_ode_ = sol.u[end].x

    # Wrap the NHS handler to avoid counting allocations in the NHS update.
    p = sol.prob.p
    p = TrixiParticles.@set p.semi = copy_semi_with_no_update_nhs(p.semi)

    if !isnothing(split_integration)
        # Mock the integrator to initialize the split integration data.
        # Note that we already replaced the neighborhood search in `p.semi` and set the
        # parallelization backend to `SerialBackend()`, so the split integration will
        # also use the modified neighborhood search and `SerialBackend()`.
        #
        # Note that doing split integration will always cause allocations through
        # OrdinaryDiffEq. However, we call `kick!` twice with the same `t` below,
        # so in the second call, where the allocations are counted, the split integration
        # will exit before the `solve!` call because the split integrator is already
        # at the time `t`.
        mock_integrator = MockIntegrator(p, (; nreject=0))
        TrixiParticles.initialize_split_integration!(split_integration, sol.u[end],
                                                     sol.t[end], mock_integrator)
        p = mock_integrator.p
    end

    # Make sure we don't use `ThreadedBroadcastArray`s here, which would cause allocations
    v_ode = Array(v_ode_)
    u_ode = Array(u_ode_)
    dv_ode = similar(v_ode)
    du_ode = similar(u_ode)

    # Wrap the NHS handler to avoid counting allocations in the NHS update.
    # Note that `p.split_integration_data.integrator.p.semi_large` is still the old
    # semidiscretization with the original NHS.
    # However, we call `kick!` twice with the same `t` below, so in the second call,
    # where the allocations are counted, the split integration does nothing because
    # the split integrator is already at the time `t`.

    try
        # Disable timers, which cause extra allocations
        TrixiParticles.disable_debug_timings()

        # We need `@invokelatest` here to ensure that the most recent method of
        # `TrixiParticles.timeit_debug_enabled()` is called, which is redefined in
        # `disable_debug_timings` above.
        return @invokelatest count_rhs_allocations_inner(dv_ode, du_ode, v_ode, u_ode,
                                                         p, t)
    finally
        # Enable timers again
        @invokelatest TrixiParticles.enable_debug_timings()
    end
end

# Function barrier to avoid type instabilites with `p`, which will cause extra allocations.
@inline function count_rhs_allocations_inner(dv_ode, du_ode, v_ode, u_ode, p, t)
    # Run RHS once to avoid counting allocations from compilation
    TrixiParticles.kick!(dv_ode, v_ode, u_ode, p, t)
    TrixiParticles.drift!(du_ode, v_ode, u_ode, p, t)

    # Count allocations
    allocations_kick = @allocated TrixiParticles.kick!(dv_ode, v_ode, u_ode, p, t)
    allocations_drift = @allocated TrixiParticles.drift!(du_ode, v_ode, u_ode, p, t)

    return allocations_kick + allocations_drift
end

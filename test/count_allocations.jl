# Wrapper for any neighborhood search that forwards `foreach_point_neighbor` to the wrapped
# neighborhood search, but doesn't do anything in the update step.
# This is used in the example tests to test for zero allocations in the `kick!` function.
struct NoUpdateNeighborhoodSearch{NHS}
    nhs::NHS
end

@inline Base.ndims(nhs::NoUpdateNeighborhoodSearch) = ndims(nhs.nhs)

# Copy a `Semidiscretization`, but wrap the neighborhood searches with
# `NoUpdateNeighborhoodSearch`.
function copy_semi_with_no_update_nhs(semi)
    neighborhood_searches = Tuple(Tuple(NoUpdateNeighborhoodSearch(nhs)
                                        for nhs in searches)
                                  for searches in semi.neighborhood_searches)

    return Semidiscretization(semi.systems, semi.ranges_u, semi.ranges_v,
                              neighborhood_searches, SerialBackend(), Ref(true), Ref(true))
end

# Forward `foreach_neighbor` to wrapped neighborhood search
@inline function PointNeighbors.foreach_neighbor(f, system_coords, neighbor_coords,
                                                 neighborhood_search::NoUpdateNeighborhoodSearch,
                                                 particle;
                                                 search_radius=PointNeighbors.search_radius(neighborhood_search.nhs))
    PointNeighbors.foreach_neighbor(f, system_coords, neighbor_coords,
                                    neighborhood_search.nhs, particle,
                                    search_radius=search_radius)
end

# No update
@inline function PointNeighbors.update!(search::NoUpdateNeighborhoodSearch, x, y;
                                        points_moving=(true, true),
                                        eachindex_y=eachindex(y),
                                        parallelization_backend=SerialBackend())
    return search
end

# Count allocations of one call to the right-hand side (`kick!` + `drift!`)
function count_rhs_allocations(sol, semi)
    t = sol.t[end]
    v_ode_, u_ode_ = sol.u[end].x

    # Make sure we don't use `ThreadedBroadcastArray`s here, which would cause allocations
    v_ode = Array(v_ode_)
    u_ode = Array(u_ode_)
    dv_ode = similar(v_ode)
    du_ode = similar(u_ode)

    # Wrap neighborhood searches to avoid counting alloctations in the NHS update
    semi_no_nhs_update = copy_semi_with_no_update_nhs(semi)

    try
        # Disable timers, which cause extra allocations
        TrixiParticles.disable_debug_timings()

        # We need `@invokelatest` here to ensure that the most recent method of
        # `TrixiParticles.timeit_debug_enabled()` is called, which is redefined in
        # `disable_debug_timings` above.
        return @invokelatest count_rhs_allocations_inner(dv_ode, du_ode, v_ode, u_ode,
                                                         semi_no_nhs_update, t)
    finally
        # Enable timers again
        @invokelatest TrixiParticles.enable_debug_timings()
    end
end

# Function barrier to avoid type instabilites with `semi_no_nhs_update`, which will
# cause extra allocations.
@inline function count_rhs_allocations_inner(dv_ode, du_ode, v_ode, u_ode,
                                             semi_no_nhs_update, t)
    # Run RHS once to avoid counting allocations from compilation
    TrixiParticles.kick!(dv_ode, v_ode, u_ode, semi_no_nhs_update, t)
    TrixiParticles.drift!(du_ode, v_ode, u_ode, semi_no_nhs_update, t)

    # Count allocations
    allocations_kick = @allocated TrixiParticles.kick!(dv_ode, v_ode, u_ode,
                                                       semi_no_nhs_update, t)
    allocations_drift = @allocated TrixiParticles.drift!(du_ode, v_ode, u_ode,
                                                         semi_no_nhs_update, t)

    return allocations_kick + allocations_drift
end

# Wrapper for any neighborhood search that forwards `for_particle_neighbor` to the wrapped
# neighborhood search, but doesn't do anything in the update step.
# This is used in the example tests to test for zero allocations in the `kick!` function.
struct NoUpdateNeighborhoodSearch{NHS}
    nhs::NHS
end

# Copy a `Semidiscretization`, but wrap the neighborhood searches with
# `NoUpdateNeighborhoodSearch`.
function copy_semi_with_no_update_nhs(semi)
    neighborhood_searches = Tuple(Tuple(NoUpdateNeighborhoodSearch(nhs)
                                        for nhs in searches)
                                  for searches in semi.neighborhood_searches)

    return Semidiscretization(semi.systems, semi.ranges_u, semi.ranges_v,
                              neighborhood_searches)
end

# Forward `for_particle_neighbor` to wrapped neighborhood search
@inline function TrixiParticles.for_particle_neighbor(f, system_coords, neighbor_coords,
                                                      neighborhood_search::NoUpdateNeighborhoodSearch;
                                                      particles=axes(system_coords, 2),
                                                      parallel=true)
    TrixiParticles.for_particle_neighbor(f, system_coords, neighbor_coords,
                                         neighborhood_search.nhs,
                                         particles=particles, parallel=parallel)
end

# No update
@inline TrixiParticles.update!(search::NoUpdateNeighborhoodSearch, coords_fun) = search

# Count allocations of one call to the right-hand side (`kick!` + `drift!`)
function count_rhs_allocations(sol, semi)
    t = sol.t[end]
    v_ode, u_ode = sol.u[end].x
    dv_ode = similar(v_ode)
    du_ode = similar(u_ode)

    # Wrap neighborhood searches to avoid counting alloctations in the NHS update
    semi_no_nhs_update = copy_semi_with_no_update_nhs(semi)

    try
        # Disable timers, which cause extra allocations
        TrixiParticles.TimerOutputs.disable_debug_timings(TrixiParticles)

        # Disable multithreading, which causes extra allocations
        return disable_polyester_threads() do
            # We need `@invokelatest` here to ensure that the most recent method of
            # `TrixiParticles.timeit_debug_enabled()` is called, which is redefined in
            # `disable_debug_timings` above.
            return @invokelatest count_rhs_allocations_inner(dv_ode, du_ode, v_ode, u_ode,
                                                         semi_no_nhs_update, t)
        end
    finally
        # Enable timers again
        @invokelatest TrixiParticles.TimerOutputs.enable_debug_timings(TrixiParticles)
    end
end

# Function barrier to avoid type instabilites with `semi_no_nhs_update`, which will
# cause extra allocations.
@inline function count_rhs_allocations_inner(dv_ode, du_ode, v_ode, u_ode, semi_no_nhs_update,
                                         t)
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

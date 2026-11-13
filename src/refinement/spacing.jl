# Update the target spacing of all particles that are not fixed by a refinement criterion
# (Alg. 1 in Haftu et al., `UpdateSpacing` in their code).
# The target spacing persists between refinement calls, so the new spacing is computed from
# the spacing of the previous call. This way, the refinement band around the fixed particles
# grows by one neighborhood per call.
@inline update_target_spacing!(system, v_ode, u_ode, semi) = system

@inline function update_target_spacing!(system::AbstractFluidSystem, v_ode, u_ode, semi)
    return update_target_spacing!(system, get_refinement(system), v_ode, u_ode, semi)
end

@inline update_target_spacing!(system, ::Nothing, v_ode, u_ode, semi) = system

function update_target_spacing!(system, refinement, v_ode, u_ode, semi)
    (; spacing_ratio, max_spacing) = refinement
    (; target_spacing, target_spacing_tmp, is_fixed) = system.cache

    u = wrap_u(u_ode, system, semi)
    system_coords = current_coordinates(u, system)

    # Write to a temporary array, so that all particles use the spacing of the previous call
    @threaded semi for particle in each_integrated_particle(system)
        if is_fixed[particle]
            target_spacing_tmp[particle] = target_spacing[particle]
        else
            target_spacing_tmp[particle] = updated_target_spacing(system, system_coords,
                                                                  semi, particle,
                                                                  spacing_ratio,
                                                                  max_spacing)
        end
    end

    @threaded semi for particle in each_integrated_particle(system)
        target_spacing[particle] = target_spacing_tmp[particle]
    end

    return system
end

@inline function updated_target_spacing(system, system_coords, semi, particle,
                                        spacing_ratio, max_spacing)
    spacing_min, spacing_max, spacing_avg,
    n_neighbors = neighborhood_spacing(system, system_coords, semi, particle)

    # Without neighbors, the spacing is not changed
    n_neighbors == 0 && return system.cache.target_spacing[particle]

    # In a uniform neighborhood, the spacing increases by `spacing_ratio` up to the global
    # `max_spacing`. Otherwise, a uniform region could never coarsen.
    if spacing_max / spacing_min < spacing_ratio^3
        return min(max_spacing, spacing_ratio * spacing_min)
    end

    return spacing_avg
end

# Minimum, maximum and geometric mean of the target spacing of the neighbors of `particle`
# within its compact support. As in the code of Haftu et al., the neighbors include the
# particle itself and only the particles of the same system are considered.
# Boundaries enter the spacing update through the particles that are fixed near them.
@inline function neighborhood_spacing(system, system_coords, semi, particle)
    (; target_spacing) = system.cache
    ELTYPE = eltype(system)

    radius = compact_support(system_smoothing_kernel(system),
                             smoothing_length(system, particle))
    neighborhood_search = get_neighborhood_search(system, semi)

    # Minimum, maximum, sum of the logarithms and number of neighbors
    init = (typemax(ELTYPE), zero(ELTYPE), zero(ELTYPE), 0)
    op(a, b) = (min(a[1], b[1]), max(a[2], b[2]), a[3] + b[3], a[4] + b[4])

    spacing_min, spacing_max, log_spacing_sum,
    n_neighbors = mapreduce_neighbor(op, system_coords, system_coords,
                                     neighborhood_search, semi.parallelization_backend,
                                     particle; init) do particle, neighbor, _, distance
        distance > radius && return init

        spacing = target_spacing[neighbor]

        return (spacing, spacing, log(spacing), 1)
    end

    # Geometric mean as in the code of Haftu et al. Summing the logarithms instead of
    # multiplying the spacings avoids an underflow of the product for many neighbors.
    spacing_avg = exp(log_spacing_sum / max(n_neighbors, 1))

    return spacing_min, spacing_max, spacing_avg, n_neighbors
end

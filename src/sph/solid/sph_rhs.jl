function rhs!(du, u, semi::SPHSolidSemidiscretization, t)
    rhs_solid!(du, u, semi, t)
end

function rhs_solid!(du, u, semi, t)
    @unpack smoothing_kernel, smoothing_length,
            boundary_conditions, gravity,
            neighborhood_search, cache = semi
    @unpack initial_coordinates, correction_matrix = cache

    @pixie_timeit timer() "rhs!" begin
        # Reset du
        @pixie_timeit timer() "reset ∂u/∂t" reset_du!(du)

        # u[1:3] = coordinates
        # u[4:6] = velocity
        @pixie_timeit timer() "main loop" @threaded for particle in eachparticle(semi)
            # dr = v
            for i in 1:ndims(semi)
                du[i, particle] = u[i + ndims(semi), particle]
            end

            pk1_particle = pk1_stress_tensor(u, particle, semi)
            pk1_particle_corrected = pk1_particle * view(correction_matrix, :, :, particle)

            # Everything here is done in the initial coordinates (except for the stress tensor)
            initial_particle_coords = get_particle_coords(initial_coordinates, semi, particle)
            for neighbor in eachneighbor(particle, initial_coordinates, neighborhood_search, semi)
                initial_neighbor_coords = get_particle_coords(initial_coordinates, semi, neighbor)

                initial_pos_diff = initial_particle_coords - initial_neighbor_coords
                initial_distance = norm(initial_pos_diff)

                if eps() < initial_distance <= compact_support(smoothing_kernel, smoothing_length)
                    pk1_neighbor = pk1_stress_tensor(u, neighbor, semi)
                    pk1_neighbor_corrected = pk1_neighbor * view(correction_matrix, :, :, neighbor)

                    calc_dv!(du, u, particle, neighbor, initial_pos_diff, initial_distance,
                             pk1_particle_corrected, pk1_neighbor_corrected, semi)
                end
            end

            calc_gravity!(du, particle, semi)
        end
    end

    return du
end


@inline function calc_dv!(du, u, particle, neighbor, initial_pos_diff, initial_distance,
                          pk1_particle_corrected, pk1_neighbor_corrected, semi)
    @unpack smoothing_kernel, smoothing_length, density_calculator, cache = semi
    @unpack mass = cache

    density_particle = get_particle_density(u, cache, density_calculator, particle)
    density_neighbor = get_particle_density(u, cache, density_calculator, neighbor)

    grad_kernel = kernel_deriv(smoothing_kernel, initial_distance, smoothing_length) *
        initial_pos_diff / initial_distance

    m_b = mass[neighbor]

    dv = m_b * (pk1_particle_corrected / density_particle^2 +
                pk1_neighbor_corrected / density_neighbor^2) * grad_kernel

    for i in 1:ndims(semi)
        du[ndims(semi) + i, particle] += dv[i]
    end

    return du
end

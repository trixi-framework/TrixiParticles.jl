function rhs!(du, u, semi::SPHSolidSemidiscretization, t)
    rhs_solid!(du, u, semi, t)
end

function rhs_solid!(du, u, semi, t)
    @unpack smoothing_kernel, smoothing_length, gravity,
            neighborhood_search, cache = semi
    @unpack initial_coordinates, current_coordinates, correction_matrix = cache

    @pixie_timeit timer() "rhs!" begin
        # Reset du
        @pixie_timeit timer() "reset ∂u/∂t" reset_du!(du)

        # Update current coordinates
        @pixie_timeit timer() "update current coordinates" update_current_coordinates(u, semi)

        # u[1:3] = coordinates
        # u[4:6] = velocity
        @pixie_timeit timer() "main loop" @threaded for particle in each_moving_particle(u, semi)
            # dr = v
            for i in 1:ndims(semi)
                du[i, particle] = u[i + ndims(semi), particle]
            end

            pk1_particle = pk1_stress_tensor(current_coordinates, particle, semi)
            pk1_particle_corrected = pk1_particle * view(correction_matrix, :, :, particle)

            # Everything here is done in the initial coordinates (except for the stress tensor)
            initial_particle_coords = get_particle_coords(initial_coordinates, semi, particle)
            for neighbor in eachneighbor(particle, initial_coordinates, neighborhood_search, semi)
                initial_neighbor_coords = get_particle_coords(initial_coordinates, semi, neighbor)

                initial_pos_diff = initial_particle_coords - initial_neighbor_coords
                initial_distance = norm(initial_pos_diff)

                if eps() < initial_distance <= compact_support(smoothing_kernel, smoothing_length)
                    pk1_neighbor = pk1_stress_tensor(current_coordinates, neighbor, semi)
                    pk1_neighbor_corrected = pk1_neighbor * view(correction_matrix, :, :, neighbor)

                    calc_dv!(du, particle, neighbor, initial_pos_diff, initial_distance,
                             pk1_particle_corrected, pk1_neighbor_corrected, semi)
                end
            end

            calc_gravity!(du, particle, semi)
        end
    end

    return du
end


@inline function update_current_coordinates(u, semi)
    @unpack cache = semi
    @unpack current_coordinates = cache

    for particle in each_moving_particle(u, semi)
        for i in 1:ndims(semi)
            current_coordinates[i, particle] = u[i, particle]
        end
    end
end


@inline function calc_dv!(du, particle, neighbor, initial_pos_diff, initial_distance,
                          pk1_particle_corrected, pk1_neighbor_corrected, semi)
    @unpack smoothing_kernel, smoothing_length, cache = semi
    @unpack mass, solid_density = cache

    density_particle = solid_density[particle]
    density_neighbor = solid_density[neighbor]

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

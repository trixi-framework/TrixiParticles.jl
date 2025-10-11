function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   system::ParticlePackingSystem{<:Any, false},
                   neighbor_system::ParticlePackingSystem, semi)
    system_coords = current_coordinates(u_particle_system, system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff
    foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords,
                           semi) do particle, neighbor, pos_diff, distance
        # Only consider particles with a distance > 0. See `src/general/smoothing_kernels.jl` for more details.
        distance^2 < eps(initial_smoothing_length(system)^2) && return

        rho_a = system.initial_condition.density[particle]
        rho_b = neighbor_system.initial_condition.density[neighbor]

        m_a = system.initial_condition.mass[particle]
        m_b = neighbor_system.initial_condition.mass[neighbor]

        V_a = m_a / rho_a
        V_b = m_b / rho_b

        p_b = system.background_pressure

        grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)

        # This vanishes for uniform particle distributions
        dv_repulsive_pressure = -(2 / m_a) * V_a * V_b * p_b * grad_kernel

        for i in 1:ndims(system)
            dv[i, particle] += dv_repulsive_pressure[i]
        end
    end

    return dv
end

# Skip for fixed systems
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   system::ParticlePackingSystem{<:Any, true}, neighbor_system, semi)
    return dv
end

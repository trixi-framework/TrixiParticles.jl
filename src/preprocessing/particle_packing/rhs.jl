function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   system::ParticlePackingSystem{<:Any, false},
                   neighbor_system::ParticlePackingSystem, semi)
    system_coords = current_coordinates(u_particle_system, system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # For `distance == 0`, the analytical gradient is zero, but the unsafe gradient
    # and the density diffusion divide by zero.
    # To account for rounding errors, we check if `distance` is almost zero.
    # Since the coordinates are in the order of the smoothing length `h`, `distance^2` is in
    # the order of `h^2`, so we need to check `distance < sqrt(eps(h^2))`.
    # Note that `sqrt(eps(h^2)) != eps(h)`.
    h = initial_smoothing_length(system)
    almostzero = sqrt(eps(h^2))

    # Loop over all pairs of particles and neighbors within the kernel cutoff
    foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords,
                           semi) do particle, neighbor, pos_diff, distance
        # Skip neighbors with the same position because the kernel gradient is zero.
        # Note that `return` only exits the closure, i.e., skips the current neighbor.
        skip_zero_distance(system) && distance < almostzero && return

        # Now that we know that `distance` is not zero, we can safely call the unsafe
        # version of the kernel gradient to avoid redundant zero checks.
        grad_kernel = smoothing_kernel_grad_unsafe(system, pos_diff,
                                                   distance, particle)

        rho_a = system.initial_condition.density[particle]
        rho_b = neighbor_system.initial_condition.density[neighbor]

        m_a = system.initial_condition.mass[particle]
        m_b = neighbor_system.initial_condition.mass[neighbor]

        V_a = m_a / rho_a
        V_b = m_b / rho_b

        p_b = system.background_pressure

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

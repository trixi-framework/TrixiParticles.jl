# Fluid-fluid and fluid-boundary interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system, neighborhood_search,
                   particle_system::EntropicallyDampedSPH,
                   neighbor_system)
    @unpack sound_speed = particle_system
    viscosity = viscosity_function(neighbor_system)

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    for_particle_neighbor(particle_system, neighbor_system,
                          system_coords, neighbor_coords,
                          neighborhood_search) do particle, neighbor, pos_diff, distance
        # Only consider particles with a distance > 0.
        distance < sqrt(eps()) && return

        rho_a = particle_density(v_particle_system, particle_system, particle)
        rho_b = particle_density(v_neighbor_system, neighbor_system, neighbor)

        pressure_a = particle_pressure(v_particle_system, particle_system, particle)
        pressure_b = particle_pressure(v_neighbor_system, neighbor_system, neighbor)

        m_a = hydrodynamic_mass(particle_system, particle)
        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        volume_a = m_a / rho_a
        volume_b = m_b / rho_b
        volume_term = (volume_a^2 + volume_b^2) / m_a

        # inter-particle averaged pressure
        pressure_tilde = (rho_b * pressure_a + rho_a * pressure_b) / (rho_a + rho_b)

        grad_kernel = smoothing_kernel_grad(particle_system, pos_diff, distance)

        dv_pressure = -volume_term * pressure_tilde * grad_kernel

        dv_viscosity = viscosity(particle_system, neighbor_system,
                                 v_particle_system, v_neighbor_system,
                                 particle, neighbor, pos_diff, distance,
                                 sound_speed, m_a, m_b)

        for i in 1:ndims(particle_system)
            dv[i, particle] += dv_pressure[i] + dv_viscosity[i]
        end

        pressure_evolution!(dv, particle_system, neighbor_system,
                            v_particle_system, v_neighbor_system,
                            particle, neighbor,
                            pos_diff, distance, sound_speed, m_a, m_b, pressure_a,
                            pressure_b, rho_a, rho_b)
    end

    return dv
end

@inline function pressure_evolution!(dv, particle_system, neighbor_system,
                                     v_particle_system, v_neighbor_system,
                                     particle, neighbor, pos_diff, distance, sound_speed,
                                     m_a, m_b, pressure_a, pressure_b, rho_a, rho_b)
    @unpack smoothing_length = particle_system

    # EDAC pressure evolution
    pressure_diff = pressure_a - pressure_b

    v_diff = current_velocity(v_particle_system, particle_system, particle) -
             current_velocity(v_neighbor_system, neighbor_system, neighbor)

    grad_kernel = smoothing_kernel_grad(particle_system, pos_diff, distance)
    artificial_EOS = m_b * rho_a * sound_speed^2 * dot(v_diff, grad_kernel) / rho_b

    eta_EDAC_a = rho_a * particle_system.nu
    eta_EDAC_b = rho_b * particle_system.nu
    eta_EDAC_tilde = 2 * eta_EDAC_a * eta_EDAC_b / (eta_EDAC_a + eta_EDAC_b)

    # TODO For variable smoothing length use average smoothing length
    tmp = eta_EDAC_tilde / (distance^2 + 0.01 * smoothing_length^2)

    volume_a = m_a / rho_a
    volume_b = m_b / rho_b
    volume_term = (volume_a^2 + volume_b^2) / m_a

    damping_term = volume_term * tmp * pressure_diff * dot(grad_kernel, pos_diff)

    dv[end, particle] += artificial_EOS + damping_term

    return dv
end

@inline viscosity_function(system) = system.viscosity
@inline viscosity_function(system::BoundarySPHSystem) = system.boundary_model.viscosity

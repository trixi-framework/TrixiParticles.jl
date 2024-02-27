# Fluid-fluid and fluid-boundary interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system, neighborhood_search,
                   particle_system::EntropicallyDampedSPHSystem,
                   neighbor_system)
    (; sound_speed, density_calculator) = particle_system
    viscosity = viscosity_model(neighbor_system)

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
        rho_mean = 0.5 * (rho_a + rho_b)

        p_a = particle_pressure(v_particle_system, particle_system, particle)
        p_b = particle_pressure(v_neighbor_system, neighbor_system, neighbor)

        m_a = hydrodynamic_mass(particle_system, particle)
        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        grad_kernel = smoothing_kernel_grad(particle_system, pos_diff, distance)

        dv_pressure = pressure_acceleration(particle_system, neighbor_system, neighbor,
                                            m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff,
                                            distance, grad_kernel, 1.0,
                                            particle_system.correction)

        dv_viscosity = viscosity(particle_system, neighbor_system,
                                 v_particle_system, v_neighbor_system,
                                 particle, neighbor, pos_diff, distance,
                                 sound_speed, m_a, m_b, rho_mean)

        for i in 1:ndims(particle_system)
            dv[i, particle] += dv_pressure[i] + dv_viscosity[i]
        end

        v_diff = current_velocity(v_particle_system, particle_system, particle) -
                 current_velocity(v_neighbor_system, neighbor_system, neighbor)

        pressure_evolution!(dv, particle_system, v_diff, grad_kernel,
                            particle, pos_diff, distance, sound_speed, m_a, m_b,
                            p_a, p_b, rho_a, rho_b)

        continuity_equation!(dv, density_calculator, v_diff, particle, m_b, rho_a, rho_b,
                             particle_system, grad_kernel)
    end

    return dv
end

@inline function pressure_evolution!(dv, particle_system, v_diff, grad_kernel, particle,
                                     pos_diff, distance, sound_speed, m_a, m_b,
                                     p_a, p_b, rho_a, rho_b)
    (; smoothing_length) = particle_system

    volume_a = m_a / rho_a
    volume_b = m_b / rho_b
    volume_term = (volume_a^2 + volume_b^2) / m_a

    # EDAC pressure evolution
    pressure_diff = p_a - p_b

    # This is basically the continuity equation times `sound_speed^2`
    artificial_eos = m_b * rho_a / rho_b * sound_speed^2 * dot(v_diff, grad_kernel)

    eta_a = rho_a * particle_system.nu_edac
    eta_b = rho_b * particle_system.nu_edac
    eta_tilde = 2 * eta_a * eta_b / (eta_a + eta_b)

    # TODO For variable smoothing length use average smoothing length
    tmp = eta_tilde / (distance^2 + 0.01 * smoothing_length^2)

    # This formulation was introduced by Hu and Adams (https://doi.org/10.1016/j.jcp.2005.09.001)
    # they argued that the formulation is more flexible because of the possibility to formulate
    # different inter-particle averages or to assume different inter-particle distributions.
    # Ramachandran (2019) and Adami (2012) use this formulation also for the pressure acceleration.
    #
    # TODO: Is there a better formulation to discretize the Laplace operator?
    # Because when using this formulation for the pressure acceleration, it is not
    # energy conserving.
    #
    # This is similar to density diffusion in WCSPH
    damping_term = volume_term * tmp * pressure_diff * dot(grad_kernel, pos_diff)

    dv[end, particle] += artificial_eos + damping_term

    return dv
end

# We need a separate method for EDAC since the density is stored in `v[end-1,:]`.
@inline function continuity_equation!(dv, density_calculator::ContinuityDensity,
                                      vdiff, particle, m_b, rho_a, rho_b,
                                      particle_system::EntropicallyDampedSPHSystem,
                                      grad_kernel)
    dv[end - 1, particle] += rho_a / rho_b * m_b * dot(vdiff, grad_kernel)

    return dv
end

@inline function continuity_equation!(dv, density_calculator,
                                      vdiff, particle, m_b, rho_a, rho_b,
                                      particle_system::EntropicallyDampedSPHSystem,
                                      grad_kernel)
    return dv
end

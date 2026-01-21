# Fluid-fluid and fluid-boundary interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::EntropicallyDampedSPHSystem,
                   neighbor_system, semi)
    (; sound_speed, density_calculator, correction, nu_edac) = particle_system

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    surface_tension_a = surface_tension_model(particle_system)
    surface_tension_b = surface_tension_model(neighbor_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff
    foreach_point_neighbor(particle_system, neighbor_system,
                           system_coords, neighbor_coords, semi;
                           points=each_integrated_particle(particle_system)) do particle,
                                                                                neighbor,
                                                                                pos_diff,
                                                                                distance
        # `foreach_point_neighbor` makes sure that `particle` and `neighbor` are
        # in bounds of the respective system. For performance reasons, we use `@inbounds`
        # in this hot loop to avoid bounds checking when extracting particle quantities.
        rho_a = @inbounds current_density(v_particle_system, particle_system, particle)
        rho_b = @inbounds current_density(v_neighbor_system, neighbor_system, neighbor)

        p_a = @inbounds current_pressure(v_particle_system, particle_system, particle)
        p_b = @inbounds current_pressure(v_neighbor_system, neighbor_system, neighbor)

        # This technique by Basa et al. 2017 (10.1002/fld.1927) aims to reduce numerical
        # errors due to large pressures by subtracting the average pressure of neighboring
        # particles.
        # It results in significant improvement for EDAC, especially with TVF,
        # but not for WCSPH, according to Ramachandran & Puri (2019), Section 3.2.
        # Note that the return value is zero when not using average pressure reduction.
        p_avg = @inbounds average_pressure(particle_system, particle)

        m_a = @inbounds hydrodynamic_mass(particle_system, particle)
        m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)

        grad_kernel = smoothing_kernel_grad(particle_system, pos_diff, distance, particle)

        dv_pressure = pressure_acceleration(particle_system, neighbor_system,
                                            particle, neighbor,
                                            m_a, m_b, p_a - p_avg, p_b - p_avg, rho_a,
                                            rho_b, pos_diff, distance, grad_kernel,
                                            correction)

        dv_viscosity_ = @inbounds dv_viscosity(particle_system, neighbor_system,
                                               v_particle_system, v_neighbor_system,
                                               particle, neighbor, pos_diff, distance,
                                               sound_speed, m_a, m_b, rho_a, rho_b,
                                               grad_kernel)

        # Extra terms in the momentum equation when using a shifting technique
        dv_tvf = @inbounds dv_shifting(shifting_technique(particle_system),
                                       particle_system, neighbor_system,
                                       v_particle_system, v_neighbor_system,
                                       particle, neighbor, m_a, m_b, rho_a, rho_b,
                                       pos_diff, distance, grad_kernel, correction)

        dv_surface_tension = surface_tension_force(surface_tension_a, surface_tension_b,
                                                   particle_system, neighbor_system,
                                                   particle, neighbor, pos_diff, distance,
                                                   rho_a, rho_b, grad_kernel)

        dv_adhesion = adhesion_force(surface_tension_a, particle_system, neighbor_system,
                                     particle, neighbor, pos_diff, distance)

        dv_particle = dv_pressure + dv_viscosity_ + dv_tvf + dv_surface_tension +
                      dv_adhesion

        for i in 1:ndims(particle_system)
            @inbounds dv[i, particle] += dv_particle[i]
        end

        v_diff = current_velocity(v_particle_system, particle_system, particle) -
                 current_velocity(v_neighbor_system, neighbor_system, neighbor)

        pressure_evolution!(dv, particle_system, neighbor_system, v_diff, grad_kernel,
                            particle, neighbor, pos_diff, distance,
                            sound_speed, m_a, m_b, p_a, p_b, rho_a, rho_b, nu_edac)

        # TODO If variable smoothing_length is used, this should use the neighbor smoothing length
        # Propagate `@inbounds` to the continuity equation, which accesses particle data
        @inbounds continuity_equation!(dv, density_calculator, particle_system,
                                       neighbor_system, v_particle_system,
                                       v_neighbor_system, particle, neighbor,
                                       pos_diff, distance, m_b, rho_a, rho_b, grad_kernel)
    end

    return dv
end

@inline function pressure_evolution!(dv, particle_system, neighbor_system, v_diff,
                                     grad_kernel, particle, neighbor,
                                     pos_diff, distance, sound_speed, m_a, m_b,
                                     p_a, p_b, rho_a, rho_b, nu_edac)
    volume_a = m_a / rho_a
    volume_b = m_b / rho_b
    volume_term = (volume_a^2 + volume_b^2) / m_a

    # EDAC pressure evolution
    pressure_diff = p_a - p_b

    # This is basically the continuity equation times `sound_speed^2`
    artificial_eos = m_b * rho_a / rho_b * sound_speed^2 * dot(v_diff, grad_kernel)

    eta_a = rho_a * nu_edac
    eta_b = rho_b * nu_edac
    eta_tilde = 2 * eta_a * eta_b / (eta_a + eta_b)

    smoothing_length_average = (smoothing_length(particle_system, particle) +
                                smoothing_length(neighbor_system, neighbor)) / 2
    tmp = eta_tilde / (distance^2 + smoothing_length_average^2 / 100)

    # This formulation was introduced by Hu and Adams (2006). https://doi.org/10.1016/j.jcp.2005.09.001
    # They argued that the formulation is more flexible because of the possibility to formulate
    # different inter-particle averages or to assume different inter-particle distributions.
    # Ramachandran (2019) and Adami (2012) use this formulation also for the pressure acceleration.
    #
    # TODO: Is there a better formulation to discretize the Laplace operator?
    # Because when using this formulation for the pressure acceleration, it is not
    # energy conserving.
    # See issue: https://github.com/trixi-framework/TrixiParticles.jl/issues/394
    #
    # This is similar to density diffusion in WCSPH
    damping_term = volume_term * tmp * pressure_diff * dot(grad_kernel, pos_diff)

    # Pressure is stored in `v` right after the velocity
    dv[ndims(particle_system) + 1, particle] += artificial_eos + damping_term

    return dv
end

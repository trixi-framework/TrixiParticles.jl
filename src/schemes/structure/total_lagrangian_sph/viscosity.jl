# Unpack the neighboring systems viscosity to dispatch on the viscosity type.
# This function is only necessary to allow `nothing` as viscosity.
# Otherwise, we could just apply the viscosity as a function directly.
@propagate_inbounds function dv_viscosity_tlsph!(dv_particle, system, v_system,
                                                 particle, neighbor,
                                                 current_pos_diff, current_distance,
                                                 m_a, m_b, rho_a, rho_b, grad_kernel)
    viscosity = system.viscosity

    return dv_viscosity_tlsph!(dv_particle, viscosity, system, v_system,
                               particle, neighbor, current_pos_diff, current_distance,
                               m_a, m_b, rho_a, rho_b, grad_kernel)
end

@propagate_inbounds function dv_viscosity_tlsph!(dv_particle, viscosity, system,
                                                 v_system, particle, neighbor,
                                                 current_pos_diff, current_distance,
                                                 m_a, m_b, rho_a, rho_b, grad_kernel)
    return viscosity(dv_particle, system, v_system, particle, neighbor,
                     current_pos_diff, current_distance,
                     m_a, m_b, rho_a, rho_b, grad_kernel)
end

@inline function dv_viscosity_tlsph!(dv_particle, viscosity::Nothing, system,
                                     v_system, particle, neighbor,
                                     current_pos_diff, current_distance,
                                     m_a, m_b, rho_a, rho_b, grad_kernel)
    return zero(current_pos_diff)
end

# Applying the viscosity according to Lin et al. (2015):
# "Geometrically nonlinear analysis of two-dimensional structures using an improved
# smoothed particle hydrodynamics method"
@propagate_inbounds function (viscosity::ArtificialViscosityMonaghan)(dv_particle,
                                                                      system::TotalLagrangianSPHSystem,
                                                                      v_system,
                                                                      particle, neighbor,
                                                                      current_pos_diff,
                                                                      current_distance,
                                                                      m_a, m_b, rho_a,
                                                                      rho_b, grad_kernel)
    v_a = current_velocity(v_system, system, particle)
    v_b = current_velocity(v_system, system, neighbor)
    v_diff = v_a - v_b

    # v_ab ⋅ r_ab
    vr = dot(v_diff, current_pos_diff)

    # Monaghan 2005 p. 1741 (doi: 10.1088/0034-4885/68/8/r01):
    # "In the case of shock tube problems, it is usual to turn the viscosity on for
    # approaching particles and turn it off for receding particles. In this way, the
    # viscosity is used for shocks and not rarefactions."
    if vr < 0
        # Compute bulk modulus from Young's modulus and Poisson's ratio.
        # See the table at the end of https://en.wikipedia.org/wiki/Lam%C3%A9_parameters
        E = young_modulus(system, particle)
        K = E / (ndims(system) * (1 - 2 * poisson_ratio(system, particle)))

        # Newton–Laplace equation
        sound_speed = sqrt(K / rho_a)

        h_a = smoothing_length(system, particle)
        h_b = smoothing_length(system, neighbor)
        h = (h_a + h_b) / 2

        rho_mean = (rho_a + rho_b) / 2

        (; alpha, beta, epsilon) = viscosity
        mu = h * vr / (current_distance^2 + epsilon * h^2)
        c = sound_speed
        pi_ab = (alpha * c * mu + beta * mu^2) / rho_mean * grad_kernel

        F = deformation_gradient(system, particle)
        det_F = det(F)
        if abs(det_F) < 1.0f-9
            return dv_particle
        end
        # See eq. 26 of Lin et al. (2015)
        dv_particle[] += m_b * det_F * inv(F)' * pi_ab
    end

    return dv_particle
end

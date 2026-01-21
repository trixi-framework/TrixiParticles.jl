# Unpack the neighboring systems viscosity to dispatch on the viscosity type.
# This function is only necessary to allow `nothing` as viscosity.
# Otherwise, we could just apply the viscosity as a function directly.
@propagate_inbounds function dv_viscosity_tlsph(system, v_system, particle, neighbor,
                                                current_pos_diff, current_distance,
                                                m_a, m_b, rho_a, rho_b, grad_kernel)
    viscosity = system.viscosity

    return dv_viscosity_tlsph(viscosity, system, v_system, particle, neighbor,
                              current_pos_diff, current_distance,
                              m_a, m_b, rho_a, rho_b, grad_kernel)
end

@propagate_inbounds function dv_viscosity_tlsph(viscosity, system,
                                                v_system, particle, neighbor,
                                                current_pos_diff, current_distance,
                                                m_a, m_b, rho_a, rho_b, grad_kernel)
    return viscosity(system, v_system, particle, neighbor,
                     current_pos_diff, current_distance,
                     m_a, m_b, rho_a, rho_b, grad_kernel)
end

@inline function dv_viscosity_tlsph(viscosity::Nothing, system,
                                    v_system, particle, neighbor,
                                    current_pos_diff, current_distance,
                                    m_a, m_b, rho_a, rho_b, grad_kernel)
    return zero(current_pos_diff)
end

# Applying the viscosity according to Lin et al. (2015):
# "Geometrically nonlinear analysis of two-dimensional structures using an improved
# smoothed particle hydrodynamics method"
@propagate_inbounds function (viscosity::ArtificialViscosityMonaghan)(system::TotalLagrangianSPHSystem,
                                                                      v_system,
                                                                      particle, neighbor,
                                                                      current_pos_diff,
                                                                      current_distance,
                                                                      m_a, m_b, rho_a,
                                                                      rho_b, grad_kernel)
    rho_mean = (rho_a + rho_b) / 2

    v_a = current_velocity(v_system, system, particle)
    v_b = current_velocity(v_system, system, neighbor)
    v_diff = v_a - v_b

    smoothing_length_particle = smoothing_length(system, particle)
    smoothing_length_neighbor = smoothing_length(system, neighbor)
    smoothing_length_average = (smoothing_length_particle + smoothing_length_neighbor) / 2

    # Compute bulk modulus from Young's modulus and Poisson's ratio.
    # See the table at the end of https://en.wikipedia.org/wiki/Lam%C3%A9_parameters
    E = young_modulus(system, particle)
    K = E / (ndims(system) * (1 - 2 * poisson_ratio(system, particle)))

    # Newton–Laplace equation
    sound_speed = sqrt(K / rho_a)

    # This is not needed for `ArtificialViscosityMonaghan`
    nu_a = nu_b = 0

    pi_ab = viscosity(sound_speed, v_diff, current_pos_diff, current_distance,
                      rho_mean, rho_a, rho_b, smoothing_length_average,
                      grad_kernel, nu_a, nu_b)

    # See eq. 26 of Lin et al. (2015)
    F = deformation_gradient(system, particle)

    if abs(det(F)) < 1.0f-9
        return zero(grad_kernel)
    end

    return m_b * det(F) * inv(F)' * pi_ab
end

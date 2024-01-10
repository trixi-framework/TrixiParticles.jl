# As shown in "Variational and momentum preservation aspects of Smooth Particle Hydrodynamic
# formulations" by Bonet and Lok (1999), for a consistent formulation this form has to be
# used with `SummationDensity`.
# This can also be seen in the tests for total energy conservation, which fail with the
# other `pressure_acceleration` form.
# We assume symmetry of the kernel gradient in this formulation. See below for the
# asymmetric version.
@inline function symmetric_pressure_acceleration_summation_density(m_a, m_b, rho_a, rho_b,
                                                                   p_a, p_b, W_a)
    return -m_b * (p_a / rho_a^2 + p_b / rho_b^2) * W_a
end

# As shown in "Variational and momentum preservation aspects of Smooth Particle Hydrodynamic
# formulations" by Bonet and Lok (1999), for a consistent formulation this form has to be
# used with `ContinuityDensity` with the formulation `\rho_a * \sum m_b / \rho_b ...`.
# This can also be seen in the tests for total energy conservation, which fail with the
# other `pressure_acceleration` form.
# We assume symmetry of the kernel gradient in this formulation. See below for the
# asymmetric version.
@inline function symmetric_pressure_acceleration_continuity_density(m_a, m_b, rho_a, rho_b,
                                                                    p_a, p_b, W_a)
    return -m_b * (p_a + p_b) / (rho_a * rho_b) * W_a
end

# Same as above, but not assuming symmetry of the kernel gradient. To be used with
# corrections that do not produce a symmetric kernel gradient.
@inline function asymmetric_pressure_acceleration_summation_density(m_a, m_b, rho_a, rho_b,
                                                                    p_a, p_b, W_a, W_b)
    return -m_b * (p_a / rho_a^2 * W_a - p_b / rho_b^2 * W_b)
end

# Same as above, but not assuming symmetry of the kernel gradient. To be used with
# corrections that do not produce a symmetric kernel gradient.
@inline function asymmetric_pressure_acceleration_continuity_density(m_a, m_b, rho_a, rho_b,
                                                                     p_a, p_b, W_a, W_b)
    return -m_b / (rho_a * rho_b) * (p_a * W_a - p_b * W_b)
end

@inline function interparticle_averaged_pressure_acceleration(m_a, m_b, rho_a, rho_b,
                                                              p_a, p_b, W_a)
    volume_a = m_a / rho_a
    volume_b = m_b / rho_b
    volume_term = (volume_a^2 + volume_b^2) / m_a

    pressure_tilde = (rho_b * p_a + rho_a * p_b) / (rho_a + rho_b)

    return -volume_term * pressure_tilde * W_a
end

function set_pressure_acceleration(pressure_acceleration,
                                   density_calculator, correction)
    return pressure_acceleration
end

function set_pressure_acceleration(pressure_acceleration::Nothing,
                                   density_calculator::SummationDensity,
                                   correction)
    return symmetric_pressure_acceleration_summation_density
end

function set_pressure_acceleration(pressure_acceleration::Nothing,
                                   density_calculator::ContinuityDensity,
                                   correction)
    return symmetric_pressure_acceleration_continuity_density
end

function set_pressure_acceleration(pressure_acceleration::Nothing,
                                   density_calculator::SummationDensity,
                                   correction::Union{KernelCorrection,
                                                     GradientCorrection,
                                                     BlendedGradientCorrection,
                                                     MixedKernelGradientCorrection})
    return asymmetric_pressure_acceleration_summation_density
end

function set_pressure_acceleration(pressure_acceleration::Nothing,
                                   density_calculator::ContinuityDensity,
                                   correction::Union{KernelCorrection,
                                                     GradientCorrection,
                                                     BlendedGradientCorrection,
                                                     MixedKernelGradientCorrection})
    return asymmetric_pressure_acceleration_continuity_density
end

# ==== Fluid System

# No correction
@inline function pressure_acceleration(pressure_correction, m_a, m_b, p_a, p_b,
                                       rho_a, rho_b, pos_diff, distance,
                                       W_a, particle_system, neighbor,
                                       neighbor_system::FluidSystem,
                                       correction)
    (; pressure_gradient) = neighbor_system

    # Without correction, the kernel gradient is symmetric, so call the symmetric
    # pressure acceleration formulation corresponding to the density calculator.
    return pressure_gradient(m_a, m_b, rho_a, rho_b, p_a, p_b, W_a)
end

# Correction
@inline function pressure_acceleration(pressure_correction, m_a, m_b, p_a, p_b,
                                       rho_a, rho_b, pos_diff, distance,
                                       W_a, particle_system, neighbor,
                                       neighbor_system::FluidSystem,
                                       correction::Union{KernelCorrection,
                                                         GradientCorrection,
                                                         BlendedGradientCorrection,
                                                         MixedKernelGradientCorrection})
    (; pressure_gradient, smoothing_kernel, smoothing_length) = neighbor_system

    W_b = smoothing_kernel_grad(neighbor_system, -pos_diff, distance, neighbor)

    # With correction, the kernel gradient is not necessarily symmetric, so call the
    # asymmetric pressure acceleration formulation corresponding to the density calculator.
    return pressure_gradient(m_a, m_b, rho_a, rho_b, p_a, p_b, W_a, W_b) *
           pressure_correction
end

# ==== Boundary and Solid System

# No correction
@inline function pressure_acceleration(pressure_correction, m_a, m_b, p_a, p_b,
                                       rho_a, rho_b, pos_diff, distance, W_a,
                                       particle_system, neighbor,
                                       neighbor_system::Union{BoundarySystem, SolidSystem},
                                       correction)
    (; boundary_model) = neighbor_system
    (; smoothing_length, pressure_gradient) = particle_system

    # Without correction, the kernel gradient is symmetric, so call the symmetric
    # pressure acceleration formulation corresponding to the density calculator.
    return pressure_acceleration(pressure_correction, m_a, m_b, p_a, p_b,
                                 rho_a, rho_b, pos_diff, distance,
                                 smoothing_length, W_a,
                                 boundary_model, pressure_gradient)
end

# Correction
@inline function pressure_acceleration(pressure_correction, m_a, m_b, p_a, p_b,
                                       rho_a, rho_b, pos_diff, distance, W_a,
                                       particle_system, neighbor,
                                       neighbor_system::Union{BoundarySystem, SolidSystem},
                                       correction::Union{KernelCorrection,
                                                         GradientCorrection,
                                                         BlendedGradientCorrection,
                                                         MixedKernelGradientCorrection})
    (; boundary_model) = neighbor_system
    (; smoothing_length, pressure_gradient) = particle_system
    W_b = smoothing_kernel_grad(neighbor_system, -pos_diff, distance, neighbor)

    # With correction, the kernel gradient is not necessarily symmetric, so call the
    # asymmetric pressure acceleration formulation corresponding to the density calculator.
    return pressure_acceleration(pressure_correction, m_a, m_b, p_a, p_b,
                                 rho_a, rho_b, pos_diff, distance,
                                 smoothing_length, W_a, W_b,
                                 boundary_model, pressure_gradient)
end

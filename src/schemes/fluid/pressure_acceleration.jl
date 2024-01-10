struct SymmetricPressureAccSummationDensity end

struct SymmetricPressureAccContinuityDensity end

struct AsymmetricPressureAccSummationDensity end

struct AsymmetricPressureAccContinuityDensity end

struct InterParticleAveragedPressureAcc end

function (grad_pressure::SymmetricPressureAccSummationDensity)(m_a, m_b, rho_a, rho_b,
                                                               p_a, p_b, W_a)
    return -m_b * (p_a / rho_a^2 + p_b / rho_b^2) * W_a
end

function (grad_pressure::SymmetricPressureAccContinuityDensity)(m_a, m_b, rho_a, rho_b,
                                                                p_a, p_b, W_a)
    return -m_b * (p_a + p_b) / (rho_a * rho_b) * W_a
end

function (grad_pressure::AsymmetricPressureAccSummationDensity)(m_a, m_b, rho_a, rho_b,
                                                                p_a, p_b, W_a, W_b)
    return -m_b * (p_a / rho_a^2 * W_a - p_b / rho_b^2 * W_b)
end

function (grad_pressure::AsymmetricPressureAccContinuityDensity)(m_a, m_b, rho_a, rho_b,
                                                                 p_a, p_b, W_a, W_b)
    return -m_b / (rho_a * rho_b) * (p_a * W_a - p_b * W_b)
end

function (grad_pressure::InterParticleAveragedPressureAcc)(m_a, m_b, rho_a, rho_b,
                                                           p_a, p_b, W_a, W_b)
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
                                   correction::Nothing)
    return SymmetricPressureAccSummationDensity()
end

function set_pressure_acceleration(pressure_acceleration::Nothing,
                                   density_calculator::ContinuityDensity,
                                   correction::Nothing)
    return SymmetricPressureAccContinuityDensity()
end

function set_pressure_acceleration(pressure_acceleration::Nothing,
                                   density_calculator::ContinuityDensity,
                                   correction::Union{KernelCorrection,
                                                     GradientCorrection,
                                                     BlendedGradientCorrection,
                                                     MixedKernelGradientCorrection})
    return AsymmetricPressureAccSummationDensity()
end

function set_pressure_acceleration(pressure_acceleration::Nothing,
                                   density_calculator::SummationDensity,
                                   correction::Union{KernelCorrection,
                                                     GradientCorrection,
                                                     BlendedGradientCorrection,
                                                     MixedKernelGradientCorrection})
    return AsymmetricPressureAccContinuityDensity()
end

@inline function pressure_acceleration(pressure_correction, m_b, p_a, p_b,
                                       rho_a, rho_b, pos_diff, distance,
                                       W_a, particle_system, neighbor,
                                       neighbor_system::FluidSystem,
                                       density_calculator,
                                       correction)
    (; grad_pressure) = neighbor_system

    return grad_pressure(m_a, m_b, rho_a, rho_b, p_a, p_b, W_a)
end

@inline function pressure_acceleration(pressure_correction, m_b, p_a, p_b,
                                       rho_a, rho_b, pos_diff, distance,
                                       W_a, particle_system,
                                       neighbor_system::FluidSystem,
                                       correction::Union{KernelCorrection,
                                                         GradientCorrection,
                                                         BlendedGradientCorrection,
                                                         MixedKernelGradientCorrection})
    (; grad_pressure, smoothing_kernel, smoothing_length) = neighbor_system

    W_b = kernel_grad(smoothing_kernel, -pos_diff, distance, smoothing_length)

    return grad_pressure(m_a, m_b, rho_a, rho_b, p_a, p_b, W_a, W_b) * pressure_correction
end

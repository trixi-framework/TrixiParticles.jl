# As shown in "Variational and momentum preservation aspects of Smooth Particle Hydrodynamic
# formulations" by Bonet and Lok (1999), for a consistent formulation this form has to be
# used with `SummationDensity`.
# This can also be seen in the tests for total energy conservation, which fail with the
# other `pressure_acceleration` form.
# We assume symmetry of the kernel gradient in this formulation. See below for the
# asymmetric version.
@inline function pressure_acceleration_summation_density(m_a, m_b, rho_a, rho_b, p_a, p_b,
                                                         W_a)
    return -m_b * (p_a / rho_a^2 + p_b / rho_b^2) * W_a
end

# Same as above, but not assuming symmetry of the kernel gradient. To be used with
# corrections that do not produce a symmetric kernel gradient.
@inline function pressure_acceleration_summation_density(m_a, m_b, rho_a, rho_b, p_a, p_b,
                                                         W_a, W_b)
    return -m_b * (p_a / rho_a^2 * W_a - p_b / rho_b^2 * W_b)
end

# As shown in "Variational and momentum preservation aspects of Smooth Particle Hydrodynamic
# formulations" by Bonet and Lok (1999), for a consistent formulation this form has to be
# used with `ContinuityDensity` with the formulation `\rho_a * \sum m_b / \rho_b ...`.
# This can also be seen in the tests for total energy conservation, which fail with the
# other `pressure_acceleration` form.
# We assume symmetry of the kernel gradient in this formulation. See below for the
# asymmetric version.
@inline function pressure_acceleration_continuity_density(m_a, m_b, rho_a, rho_b, p_a, p_b,
                                                          W_a)
    return -m_b * (p_a + p_b) / (rho_a * rho_b) * W_a
end

# Same as above, but not assuming symmetry of the kernel gradient. To be used with
# corrections that do not produce a symmetric kernel gradient.
@inline function pressure_acceleration_continuity_density(m_a, m_b, rho_a, rho_b, p_a, p_b,
                                                          W_a, W_b)
    return -m_b / (rho_a * rho_b) * (p_a * W_a - p_b * W_b)
end

function choose_pressure_acceleration_formulation(pressure_acceleration,
                                                  density_calculator, NDIMS, ELTYPE,
                                                  correction)
    if correction isa KernelCorrection ||
       correction isa GradientCorrection ||
       correction isa BlendedGradientCorrection ||
       correction isa MixedKernelGradientCorrection
        if isempty(methods(pressure_acceleration,
                           (ELTYPE, ELTYPE, ELTYPE, ELTYPE, ELTYPE, ELTYPE,
                            SVector{NDIMS, ELTYPE}, SVector{NDIMS, ELTYPE})))
            throw(ArgumentError("when a correction with an asymmetric kernel gradient is " *
                                "used, the passed pressure acceleration formulation must " *
                                "provide a version with the arguments " *
                                "`m_a, m_b, rho_a, rho_b, p_a, p_b, W_a, W_b`"))
        end
    else
        if isempty(methods(pressure_acceleration,
                           (ELTYPE, ELTYPE, ELTYPE, ELTYPE, ELTYPE, ELTYPE,
                            SVector{NDIMS, ELTYPE})))
            throw(ArgumentError("when not using a correction with an asymmetric kernel " *
                                "gradient, the passed pressure acceleration formulation must " *
                                "provide a version with the arguments " *
                                "`m_a, m_b, rho_a, rho_b, p_a, p_b, W_a`, " *
                                "using the symmetry of the kernel gradient"))
        end
    end

    return pressure_acceleration
end

function choose_pressure_acceleration_formulation(pressure_acceleration::Nothing,
                                                  density_calculator::SummationDensity,
                                                  initial_condition,
                                                  correction)

    # Choose the pressure acceleration formulation corresponding to the density calculator.
    return pressure_acceleration_summation_density
end

function choose_pressure_acceleration_formulation(pressure_acceleration::Nothing,
                                                  density_calculator::ContinuityDensity,
                                                  initial_condition,
                                                  correction)

    # Choose the pressure acceleration formulation corresponding to the density calculator.
    return pressure_acceleration_continuity_density
end

# Formulation using symmetric gradient formulation for corrections not depending on local neighborhood.
@inline function pressure_acceleration(particle_system, neighbor_system, neighbor,
                                       m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff,
                                       distance, W_a, pressure_correction,
                                       correction)
    (; pressure_acceleration_formulation) = particle_system

    # Without correction or with `AkinciFreeSurfaceCorrection`, the kernel gradient is
    # symmetric, so call the symmetric version of the pressure acceleration formulation.
    return pressure_acceleration_formulation(m_a, m_b, rho_a, rho_b, p_a, p_b, W_a) *
           pressure_correction
end

# Formulation using asymmetric gradient formulation for corrections depending on local neighborhood.
@inline function pressure_acceleration(particle_system, neighbor_system, neighbor,
                                       m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff,
                                       distance, W_a, pressure_correction,
                                       correction::Union{KernelCorrection,
                                                         GradientCorrection,
                                                         BlendedGradientCorrection,
                                                         MixedKernelGradientCorrection})
    (; pressure_acceleration_formulation) = particle_system

    W_b = smoothing_kernel_grad(neighbor_system, -pos_diff, distance, neighbor)

    # With correction, the kernel gradient is not necessarily symmetric, so call the
    # asymmetric version of the pressure acceleration formulation.
    return pressure_acceleration_formulation(m_a, m_b, rho_a, rho_b, p_a, p_b, W_a, W_b) *
           pressure_correction
end

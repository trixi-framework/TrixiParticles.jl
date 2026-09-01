# As shown in "Variational and momentum preservation aspects of Smooth Particle Hydrodynamic
# formulations" by Bonet and Lok (1999), for a consistent formulation this form has to be
# used with `SummationDensity`.
# This can also be seen in the tests for total energy conservation, which fail with the
# other `pressure_acceleration` form.
# We assume symmetry of the kernel gradient in this formulation. See below for the
# asymmetric version.
@inline function pressure_acceleration_summation_density(m_a, m_b, rho_a, rho_b, p_a, p_b,
                                                         W_a)
    # Since this is one of the most performance critical functions, using fast divisions
    # here gives a significant speedup on GPUs.
    # See the docs page "Development" for more details on `div_fast`.
    # -m_b * (p_a / ρ_a^2 + p_b / ρ_b^2) * ∇W_ab
    return -m_b * (div_fast(p_a, rho_a^2) + div_fast(p_b, rho_b^2)) * W_a
end

# Same as above, but not assuming symmetry of the kernel gradient. To be used with
# corrections that do not produce a symmetric kernel gradient.
@inline function pressure_acceleration_summation_density(m_a, m_b, rho_a, rho_b, p_a, p_b,
                                                         W_a, W_b)
    # Since this is one of the most performance critical functions, using fast divisions
    # here gives a significant speedup on GPUs.
    # See the docs page "Development" for more details on `div_fast`.
    return -m_b * (div_fast(p_a, rho_a^2) * W_a - div_fast(p_b, rho_b^2) * W_b)
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
    # Since this is one of the most performance critical functions, using fast divisions
    # here gives a significant speedup on GPUs.
    # See the docs page "Development" for more details on `div_fast`.
    # -m_b * (p_a + p_b) / (ρ_a * ρ_b) * ∇W_ab
    return -m_b * div_fast(p_a + p_b, rho_a * rho_b) * W_a
end

# Same as above, but not assuming symmetry of the kernel gradient. To be used with
# corrections that do not produce a symmetric kernel gradient.
@inline function pressure_acceleration_continuity_density(m_a, m_b, rho_a, rho_b, p_a, p_b,
                                                          W_a, W_b)
    # Since this is one of the most performance critical functions, using fast divisions
    # here gives a significant speedup on GPUs.
    # See the docs page "Development" for more details on `div_fast`.
    return -div_fast(m_b, rho_a * rho_b) * (p_a * W_a - p_b * W_b)
end

@doc raw"""
    tensile_instability_control

Pressure acceleration formulation to prevent tensile instability
by [Sun et al. (2018)](@cite Sun2018).

This formulation can be passed as keyword argument `pressure_acceleration` to
the [`WeaklyCompressibleSPHSystem`](@ref) constructor.
See [Tensile Instability Control](@ref tic) for more information on this technique.

!!! warning
    Tensile Instability Control needs to be disabled close to the free surface
    and therefore requires a free surface detection method. This is not yet implemented.
    **This technique cannot be used in a free surface simulation.**
"""
@inline function tensile_instability_control(m_a, m_b, rho_a, rho_b, p_a, p_b, W_a)
    # Same as `pressure_acceleration_continuity_density`, but using the minus formulation
    # when pressures are negative to avoid tensile instability.
    return -m_b * (abs(p_a) + p_b) / (rho_a * rho_b) * W_a
end

# This formulation was introduced by Hu and Adams (2006). https://doi.org/10.1016/j.jcp.2005.09.001
# They argued that the formulation is more flexible because of the possibility to formulate
# different inter-particle averages or to assume different inter-particle distributions.
# Ramachandran (2019) and Adami (2012) use this formulation for the pressure acceleration.
#
# With a symmetric radial kernel gradient, the formulation conserves linear and angular
# momentum, but not energy. With asymmetric corrected gradients, the formulation below still
# conserves linear momentum, while the generally non-central pair force does not conserve
# angular momentum.
#
# Note that the authors also used this formulation for an ISPH method in (https://doi.org/10.1016/j.jcp.2007.07.013)
#
# TODO: Further investigations on energy conservation.
@inline function inter_particle_averaged_pressure(m_a, m_b, rho_a, rho_b, p_a, p_b, W_a)
    volume_a = m_a / rho_a
    volume_b = m_b / rho_b
    volume_term = (volume_a^2 + volume_b^2) / m_a

    # Inter-particle averaged pressure
    pressure_tilde = (rho_b * p_a + rho_a * p_b) / (rho_a + rho_b)

    return -volume_term * pressure_tilde * W_a
end

# Linear-momentum-conserving extension for correction methods with asymmetric kernel gradients.
# The asymmetric overload is selected based on the configured correction, even when a particular
# particle pair happens to have symmetric gradients. It reduces to the symmetric formulation
# above when `W_b == -W_a`.
@inline function inter_particle_averaged_pressure(m_a, m_b, rho_a, rho_b, p_a, p_b, W_a,
                                                  W_b)
    volume_a = m_a / rho_a
    volume_b = m_b / rho_b
    volume_term = (volume_a^2 + volume_b^2) / m_a
    pressure_tilde = (rho_b * p_a + rho_a * p_b) / (rho_a + rho_b)

    half = oftype(volume_term, 0.5)
    return -half * volume_term * pressure_tilde * (W_a - W_b)
end

function choose_pressure_acceleration_formulation(pressure_acceleration,
                                                  density_calculator, NDIMS, ELTYPE,
                                                  correction)
    if has_asymmetric_gradient(correction)
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

@inline has_asymmetric_gradient(::Any) = false

@inline has_asymmetric_gradient(::Union{KernelCorrection,
                                        GradientCorrection,
                                        BlendedGradientCorrection,
                                        MixedKernelGradientCorrection}) = true

function choose_pressure_acceleration_formulation(pressure_acceleration::Nothing,
                                                  density_calculator::SummationDensity,
                                                  NDIMS, ELTYPE,
                                                  correction)

    # Choose the pressure acceleration formulation corresponding to the density calculator.
    return pressure_acceleration_summation_density
end

function choose_pressure_acceleration_formulation(pressure_acceleration::Nothing,
                                                  density_calculator::ContinuityDensity,
                                                  NDIMS, ELTYPE,
                                                  correction)

    # Choose the pressure acceleration formulation corresponding to the density calculator.
    return pressure_acceleration_continuity_density
end

@inline pressure_acceleration_formulation(system) = system.pressure_acceleration_formulation

# Use the local pressure offset by default. Specialized pair methods can combine offsets from
# both systems when both directed interaction evaluations support the same reduction.
@inline interaction_pressure_offset(system, particle) = zero(eltype(system))

@propagate_inbounds function pair_pressure_offset(system, neighbor_system, particle,
                                                  neighbor)
    return interaction_pressure_offset(system, particle)
end

# Formulation using both corrected gradients whenever either endpoint uses a correction that
# breaks pair-gradient symmetry. Both directed evaluations then use the same formulation.
@inline function pressure_acceleration(particle_system, neighbor_system, particle, neighbor,
                                       m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff,
                                       distance, W_a, correction)
    if has_asymmetric_gradient(correction) ||
       has_asymmetric_gradient(hydrodynamic_correction(neighbor_system))
        W_b = hydrodynamic_smoothing_kernel_grad(neighbor_system, -pos_diff, distance,
                                                 neighbor)

        return pressure_acceleration_formulation(particle_system)(m_a, m_b, rho_a, rho_b,
                                                                  p_a, p_b, W_a, W_b)
    end

    return pressure_acceleration_formulation(particle_system)(m_a, m_b, rho_a, rho_b,
                                                              p_a, p_b, W_a)
end

function check_pressure_acceleration_configuration(systems, interaction_matrix)
    foreach_system(systems) do particle_system
        particle_system isa AbstractFluidSystem || return

        foreach_system(systems) do neighbor_system
            interaction = interaction_matrix[findfirst(system -> system === particle_system,
                                                       systems),
                                             findfirst(system -> system === neighbor_system,
                                                       systems)]
            interaction === true || return

            correction = system_correction(particle_system)
            if !has_asymmetric_gradient(correction) &&
               !has_asymmetric_gradient(hydrodynamic_correction(neighbor_system))
                return
            end

            NDIMS = ndims(particle_system)
            ELTYPE = eltype(particle_system)
            formulation = pressure_acceleration_formulation(particle_system)
            signature = (ELTYPE, ELTYPE, ELTYPE, ELTYPE, ELTYPE, ELTYPE,
                         SVector{NDIMS, ELTYPE}, SVector{NDIMS, ELTYPE})
            if isempty(methods(formulation, signature))
                throw(ArgumentError("the pressure acceleration formulation of " *
                                    "`$(nameof(typeof(particle_system)))` must provide " *
                                    "`m_a, m_b, rho_a, rho_b, p_a, p_b, W_a, W_b` when " *
                                    "interacting with an asymmetric gradient correction"))
            end
        end
    end

    return nothing
end

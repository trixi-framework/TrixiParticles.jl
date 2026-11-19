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
# However, the tests show that the formulation is only linear and angular momentum conserving
# but not energy conserving.
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

function choose_pressure_acceleration_formulation(pressure_acceleration,
                                                  density_calculator, NDIMS, ELTYPE,
                                                  correction)
    if correction isa AsymmetricKernelGradientCorrection
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

# Boundary laws can specialize physical pressure without affecting the TVF operators.
@inline function pressure_acceleration(particle_system, neighbor_system, particle, neighbor,
                                       m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff,
                                       distance, W_a, correction)
    formulation = pressure_acceleration_formulation(particle_system)
    return pressure_acceleration_interparticle(formulation, neighbor_system, neighbor,
                                               m_a, m_b, p_a, p_b, rho_a, rho_b,
                                               pos_diff, distance, W_a, correction)
end

# Evaluate the inter-particle formulation without dispatching to a physical boundary
# force. TVF uses scalar unit pressures for background-pressure shifting and tensors
# for its momentum correction. Neither term should apply the radial repulsion law.
@inline function pressure_acceleration_interparticle(formulation::F, neighbor_system,
                                                     neighbor, m_a, m_b, value_a, value_b,
                                                     rho_a, rho_b, pos_diff, distance,
                                                     W_a, correction) where {F}
    # With no correction, or with a correction that leaves the kernel gradient
    # unchanged (such as `AkinciFreeSurfaceCorrection`), use the symmetric
    # formulation: it assumes W_b = -W_a and takes only W_a.
    return formulation(m_a, m_b, rho_a, rho_b, value_a, value_b, W_a)
end

@inline function pressure_acceleration_interparticle(formulation::F, neighbor_system,
                                                     neighbor, m_a, m_b, value_a, value_b,
                                                     rho_a, rho_b, pos_diff, distance,
                                                     W_a,
                                                     ::AsymmetricKernelGradientCorrection) where {F}
    # These corrections depend on the local neighborhood and can break gradient
    # antisymmetry. Call the asymmetric formulation with both gradients, evaluating
    # W_b at the neighbor with the reversed displacement.
    W_b = smoothing_kernel_grad(neighbor_system, -pos_diff, distance, neighbor)
    return formulation(m_a, m_b, rho_a, rho_b, value_a, value_b, W_a, W_b)
end

# The last argument is the kernel gradient `W_b` of the neighbor if it has already been
# computed (see `interact_structure_fluid!`), or `nothing` to compute it when required.
@inline function pressure_acceleration(particle_system, neighbor_system, particle, neighbor,
                                       m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff,
                                       distance, W_a, correction, W_b::Nothing)
    return pressure_acceleration(particle_system, neighbor_system, particle, neighbor,
                                 m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff, distance,
                                 W_a, correction)
end

# The structure RHS computes the boundary gradient once for both the pressure
# acceleration and the continuity equation of the boundary particles.
@inline function pressure_acceleration(particle_system, neighbor_system, particle, neighbor,
                                       m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff,
                                       distance, W_a, ::AsymmetricKernelGradientCorrection,
                                       W_b::AbstractVector)
    # As in the two-gradient method above, the correction can break antisymmetry.
    # Use the supplied W_b instead of evaluating the neighbor gradient again.
    return pressure_acceleration_formulation(particle_system)(m_a, m_b, rho_a, rho_b,
                                                              p_a, p_b, W_a, W_b)
end

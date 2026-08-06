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
    Direct use of this function must be disabled close to a free surface. For supported
    Morris/CSS free-surface simulations, pass
    [`InterfaceAwareTensileInstabilityControl`](@ref) as `pressure_acceleration` instead.
"""
@inline function tensile_instability_control(m_a, m_b, rho_a, rho_b, p_a, p_b, W_a)
    # Same as `pressure_acceleration_continuity_density`, but using the minus formulation
    # when pressures are negative to avoid tensile instability.
    return -m_b * (abs(p_a) + p_b) / (rho_a * rho_b) * W_a
end

"""
    InterfaceAwareTensileInstabilityControl(; strength=1.0)

Apply [`tensile_instability_control`](@ref) in fully supported fluid interiors and blend back
to the conservative continuity-density pressure acceleration across a represented free surface.
The control is disabled for fluid-boundary interactions.

This explicit opt-in requires `ContinuityDensity`, unclipped pressure, no kernel-gradient
correction, and an interface-activity-providing surface-normal method: either a
[`ColorfieldSurfaceNormal`](@ref) with Morris/CSS surface tension or a
[`CorrectedCSFSurfaceNormal`](@ref) with [`SurfaceTensionMorris`](@ref).
The `strength` in `(0, 1]` scales only the tensile correction; `1` recovers the complete
interior TIC formulation.
"""
struct InterfaceAwareTensileInstabilityControl{T <: Real}
    strength::T

    function InterfaceAwareTensileInstabilityControl(; strength=1.0)
        strength isa Real && isfinite(strength) && 0 < strength <= 1 ||
            throw(ArgumentError("`strength` must be finite and in (0, 1]"))
        new{typeof(strength)}(strength)
    end
end

@inline validate_interface_aware_tic(pressure_acceleration, density_calculator,
                                     state_equation, surface_normal_method,
                                     surface_tension, correction) = nothing

function validate_interface_aware_tic(::InterfaceAwareTensileInstabilityControl,
                                      density_calculator, state_equation,
                                      surface_normal_method, surface_tension, correction)
    density_calculator isa ContinuityDensity ||
        throw(ArgumentError("`InterfaceAwareTensileInstabilityControl` requires `ContinuityDensity`"))
    isnothing(correction) ||
        throw(ArgumentError("`InterfaceAwareTensileInstabilityControl` does not support kernel-gradient corrections"))
    supports_interface_aware_tic(surface_normal_method, surface_tension) ||
        throw(ArgumentError("`InterfaceAwareTensileInstabilityControl` requires " *
                            "`ColorfieldSurfaceNormal` with Morris or CSS surface tension " *
                            "or `CorrectedCSFSurfaceNormal` with `SurfaceTensionMorris`"))
    if !isnothing(state_equation) && clip_negative_pressure(state_equation)
        throw(ArgumentError("`InterfaceAwareTensileInstabilityControl` requires unclipped negative pressure"))
    end
    return nothing
end

@inline function interface_aware_tensile_acceleration(m_a, m_b, rho_a, rho_b, p_a, p_b,
                                                      W_a, activity_a, activity_b,
                                                      strength)
    standard = pressure_acceleration_continuity_density(m_a, m_b, rho_a, rho_b,
                                                        p_a, p_b, W_a)
    controlled = tensile_instability_control(m_a, m_b, rho_a, rho_b, p_a, p_b, W_a)
    interface_activity = clamp(max(activity_a, activity_b), zero(activity_a),
                               one(activity_a))
    interior_weight = strength * (one(interface_activity) - interface_activity)
    return standard + interior_weight * (controlled - standard)
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

function choose_pressure_acceleration_formulation(control::InterfaceAwareTensileInstabilityControl,
                                                  density_calculator, NDIMS, ELTYPE,
                                                  correction)
    density_calculator isa ContinuityDensity ||
        throw(ArgumentError("`InterfaceAwareTensileInstabilityControl` requires `ContinuityDensity`"))
    isnothing(correction) ||
        throw(ArgumentError("`InterfaceAwareTensileInstabilityControl` does not support kernel-gradient corrections"))
    return control
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

@inline function evaluate_pressure_acceleration(formulation, particle_system,
                                                neighbor_system, particle, neighbor,
                                                m_a, m_b, rho_a, rho_b, p_a, p_b, W_a)
    return formulation(m_a, m_b, rho_a, rho_b, p_a, p_b, W_a)
end

@inline function evaluate_pressure_acceleration(::InterfaceAwareTensileInstabilityControl,
                                                particle_system, neighbor_system,
                                                particle, neighbor, m_a, m_b,
                                                rho_a, rho_b, p_a, p_b, W_a)
    return pressure_acceleration_continuity_density(m_a, m_b, rho_a, rho_b,
                                                    p_a, p_b, W_a)
end

@inline function evaluate_pressure_acceleration(control::InterfaceAwareTensileInstabilityControl,
                                                particle_system,
                                                neighbor_system::AbstractFluidSystem,
                                                particle, neighbor, m_a, m_b,
                                                rho_a, rho_b, p_a, p_b, W_a)
    activity_a = surface_interface_activity(particle_system, particle)
    activity_b = surface_interface_activity(neighbor_system, neighbor)
    return interface_aware_tensile_acceleration(m_a, m_b, rho_a, rho_b, p_a, p_b, W_a,
                                                activity_a, activity_b, control.strength)
end

@inline function evaluate_pressure_acceleration(::InterfaceAwareTensileInstabilityControl,
                                                particle_system,
                                                neighbor_system::AbstractFluidSystem,
                                                particle, neighbor, m_a, m_b,
                                                rho_a, rho_b, p_a::SMatrix, p_b, W_a)
    return pressure_acceleration_continuity_density(m_a, m_b, rho_a, rho_b,
                                                    p_a, p_b, W_a)
end

# Formulation using symmetric gradient formulation for corrections not depending on local neighborhood.
@inline function pressure_acceleration(particle_system, neighbor_system, particle, neighbor,
                                       m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff,
                                       distance, W_a, correction)
    # Without correction or with `AkinciFreeSurfaceCorrection`, the kernel gradient is
    # symmetric, so call the symmetric version of the pressure acceleration formulation.
    formulation = pressure_acceleration_formulation(particle_system)
    return evaluate_pressure_acceleration(formulation, particle_system, neighbor_system,
                                          particle, neighbor, m_a, m_b, rho_a, rho_b,
                                          p_a, p_b, W_a)
end

# Formulation using asymmetric gradient formulation for corrections depending on local neighborhood.
@inline function pressure_acceleration(particle_system, neighbor_system, particle, neighbor,
                                       m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff,
                                       distance, W_a,
                                       correction::Union{KernelCorrection,
                                                         GradientCorrection,
                                                         BlendedGradientCorrection,
                                                         MixedKernelGradientCorrection})
    W_b = smoothing_kernel_grad(neighbor_system, -pos_diff, distance, neighbor)

    # With correction, the kernel gradient is not necessarily symmetric, so call the
    # asymmetric version of the pressure acceleration formulation.
    return pressure_acceleration_formulation(particle_system)(m_a, m_b, rho_a, rho_b,
                                                              p_a, p_b, W_a, W_b)
end

@doc raw"""
    PenaltyForceGanzenmueller(; alpha=0.1)

Penalty force to ensure regular particle positions under large deformations.

# Keywords
- `alpha`: Coefficient to control the amplitude of hourglass correction.

"""
struct PenaltyForceGanzenmueller{ELTYPE}
    alpha::ELTYPE
    function PenaltyForceGanzenmueller(; alpha=0.1)
        new{typeof(alpha)}(alpha)
    end
end

@inline function dv_penalty_force(penalty_force::Nothing,
                                  particle, neighbor, initial_pos_diff, initial_distance,
                                  current_pos_diff, current_distance,
                                  system, m_a, m_b, rho_a, rho_b)
    return zero(initial_pos_diff)
end

@inline function dv_penalty_force(penalty_force::PenaltyForceGanzenmueller,
                                  particle, neighbor, initial_pos_diff, initial_distance,
                                  current_pos_diff, current_distance,
                                  system, m_a, m_b, rho_a, rho_b)
    (; young_modulus) = system

    volume_a = m_a / rho_a
    volume_b = m_b / rho_b

    kernel_weight = smoothing_kernel(system, initial_distance, particle)

    J_a = deformation_gradient(system, particle)
    J_b = deformation_gradient(system, neighbor)

    # Use the symmetry of epsilon to simplify computations
    eps_sum = (J_a + J_b) * initial_pos_diff - 2 * current_pos_diff
    delta_sum = dot(eps_sum, current_pos_diff) / current_distance

    E = young_modulus_per_particle(young_modulus, particle)

    f = (penalty_force.alpha / 2) * volume_a * volume_b *
        kernel_weight / initial_distance^2 * E * delta_sum * current_pos_diff /
        current_distance

    # Divide force by mass to obtain acceleration
    return f / m_a
end

function young_modulus_per_particle(young_modulus::AbstractVector, particle)
    return young_modulus[particle]
end

function young_modulus_per_particle(young_modulus, particle)
    return young_modulus
end

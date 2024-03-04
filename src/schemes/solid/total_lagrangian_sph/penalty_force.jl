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

@inline function calc_penalty_force!(dv, particle, neighbor, initial_pos_diff,
                                     initial_distance, system, m_a, m_b, rho_a, rho_b,
                                     penalty_force::PenaltyForceGanzenmueller)
    (; young_modulus) = system

    current_pos_diff = current_coords(system, particle) -
                       current_coords(system, neighbor)
    current_distance = norm(current_pos_diff)

    volume_particle = m_a / rho_a
    volume_neighbor = m_b / rho_b

    kernel_weight = smoothing_kernel(system, initial_distance)

    J_a = deformation_gradient(system, particle)
    J_b = deformation_gradient(system, neighbor)

    # Use the symmetry of epsilon to simplify computations
    eps_sum = (J_a + J_b) * initial_pos_diff - 2 * current_pos_diff
    delta_sum = dot(eps_sum, current_pos_diff) / current_distance

    f = 0.5 * penalty_force.alpha * volume_particle * volume_neighbor *
        kernel_weight / initial_distance^2 * young_modulus * delta_sum *
        current_pos_diff / current_distance

    @inbounds for i in 1:ndims(system)
        # Divide force by mass to obtain acceleration
        dv[i, particle] += f[i] / m_a
    end

    return dv
end

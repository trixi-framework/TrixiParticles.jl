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

@inline function add_penalty_force(dv_particle, penalty_force::Nothing,
                                   particle, neighbor, initial_pos_diff, initial_distance,
                                   current_pos_diff, current_distance,
                                   system, m_a, m_b, rho_a, rho_b, F_a, F_b)
    return dv_particle
end

@propagate_inbounds function add_penalty_force(dv_particle,
                                               penalty_force::PenaltyForceGanzenmueller,
                                               particle, neighbor, initial_pos_diff,
                                               initial_distance,
                                               current_pos_diff, current_distance,
                                               system, m_a, m_b, rho_a, rho_b, F_a, F_b)
    (; alpha) = penalty_force

    # Since this is one of the most performance critical functions, using fast divisions
    # here gives a significant speedup on GPUs.
    # See the docs page "Development" for more details on `div_fast`.
    volume_a = div_fast(m_a, rho_a)
    volume_b = div_fast(m_b, rho_b)

    kernel_weight = smoothing_kernel(system, initial_distance, particle)

    E_a = young_modulus(system, particle)
    E_b = young_modulus(system, neighbor)

    # Note that this is actually ϵ^b_ab = -ϵ^b_ba in the paper, so we later compute
    # δ^b_ab instead of δ^b_ba, but δ^b_ab = δ^b_ba because of antisymmetry of x_ab and ϵ_ab.
    eps_a = F_a * initial_pos_diff - current_pos_diff
    eps_b = F_b * initial_pos_diff - current_pos_diff

    # This is (E_a * delta_a + E_b * delta_b) * current_distance.
    # Pulling the division by `current_distance` out allows us to do one division by
    # `current_distance^2` instead.
    delta_sum = E_a * dot(eps_a, current_pos_diff) + E_b * dot(eps_b, current_pos_diff)

    # The division contains all scalar factors, which are then multiplied by
    # the vector `current_pos_diff` at the end.
    # We already divide by `m_a` to obtain an acceleration.
    # Since this is one of the most performance critical functions, using fast divisions
    # here gives a significant speedup on GPUs.
    # See the docs page "Development" for more details on `div_fast`.
    # For the particle itself, `initial_distance == current_distance == 0`, and both
    # `delta_sum` and `current_pos_diff` are exactly zero.
    # To avoid branching in the hot loop, we clamp the denominator to the smallest
    # positive normal number. Then, the result is exactly zero for the particle itself.
    # We use `ifelse` instead of `max` because it is faster (see `clamp0`).
    # This is significantly faster than `clamp_distance(initial_distance^2, h^2)`.
    denominator = initial_distance^2 * current_distance^2 * m_a
    denominator_ = ifelse(denominator < floatmin(denominator), floatmin(denominator),
                          denominator)
    dv_particle += div_fast((alpha / 2) * volume_a * volume_b * kernel_weight * delta_sum,
                            denominator_) *
                   current_pos_diff

    return dv_particle
end

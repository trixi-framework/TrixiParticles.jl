struct CubicSplitting{ELTYPE}
    epsilon :: ELTYPE
    alpha   :: ELTYPE

    function CubicSplitting(; epsilon=0.5, alpha=0.5)
        new{typeof(epsilon)}(epsilon, alpha)
    end
end

function relative_positions(refinement_pattern::CubicSplitting, ::System{2},
                            particle_spacing)
    (; epsilon) = refinement_pattern

    direction_1 = normalize([1.0, 1.0])
    direction_2 = normalize([1.0, -1.0])
    direction_3 = -direction_1
    direction_4 = -direction_2

    relative_position = hcat(particle_spacing * epsilon * direction_1,
                             particle_spacing * epsilon * direction_2,
                             particle_spacing * epsilon * direction_3,
                             particle_spacing * epsilon * direction_4)

    return reinterpret(reshape, SVector{2, typeof(epsilon)}, relative_position)
end

# TODO: Clarify refinement pattern. Cubic splitting? Triangular or hexagonal?
# See https://www.sciencedirect.com/science/article/pii/S0020740319317023
@inline nchilds(system, refinement_pattern) = 2^ndims(system)

@inline smoothing_length_child(system, refinement_pattern) = system.smoothing_length

@inline mass_child(system, refinement_pattern) = system.mass

@inline particle_spacing_child(system, refinement_pattern) = system.initial_condition.particle_spacing

# ==== Refinement criteria
struct RefinementZone end # TODO

@inline function (refinement_criterion::RefinementZone)(system, particle,
                                                        v, u, v_ode, u_ode, semi)
    (; zone_origin, spanning_set) = refinement_criterion
    particle_position = current_coords(u, system, particle) - zone_origin

    for dim in 1:ndims(system)
        span_dim = spanning_set[dim]
        # Checks whether the projection of the particle position
        # falls within the range of the zone.
        if !(0 <= dot(particle_position, span_dim) <= dot(span_dim, span_dim))

            # Particle is not in refinement zone.
            return false
        end
    end

    # Particle is in refinement zone.
    return true
end

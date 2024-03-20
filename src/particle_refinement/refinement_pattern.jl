struct CubicSplitting{ELTYPE}
    epsilon :: ELTYPE
    alpha   :: ELTYPE

    function CubicSplitting(; epsilon=0.5, alpha=0.5)
        new{typeof(epsilon)}(epsilon, alpha)
    end
end

function relative_position_children(system::System{2},
                                    refinement_pattern::CubicSplitting)
    (; initial_condition) = system
    (; particle_spacing) = initial_condition
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

function mass_distribution(system::System{2}, refinement_pattern::CubicSplitting)
    lambda = 1 / nchilds(system, refinement_pattern)

    return lambda .* ones(nchilds(system, refinement_pattern))
end

# TODO: Clarify refinement pattern. Cubic splitting? Triangular or hexagonal?
# See https://www.sciencedirect.com/science/article/pii/S0020740319317023
@inline nchilds(system, rp::CubicSplitting) = 2^ndims(system)

@inline mass_child(system, mass, rp::CubicSplitting) = mass / nchilds(system, rp)

@inline smoothing_length_child(system, refinement_pattern) = refinement_pattern.alpha *
                                                             system.smoothing_length

@inline particle_spacing_child(system, refinement_pattern) = system.initial_condition.particle_spacing *
                                                             refinement_pattern.epsilon

function mass_distribution(system, refinement_pattern)
    # TODO:
    #=
    if refinement_pattern.center_particle
        # solve minimisation problem

        # For `HexagonalSplitting`, `separation_parameter=0.4` and `smoothing_ratio=0.9`:
        lambdas = fill(0.1369, (6,))
        push!(lambdas, 0.1787) # central particle

        if isapprox(sum(lambdas), 1.0, rtol=1e-4)
            return SVector(lambdas...)
        end

        error("no mass conservation")
    else
        =#
    lambda = 1 / nchilds(system, refinement_pattern)

    return fill(lambda, SVector{nchilds(system, refinement_pattern), eltype(system)})
    #end
end

struct CubicSplitting{ELTYPE}
    separation_parameter :: ELTYPE
    smoothing_ratio      :: ELTYPE
    center_particle      :: Bool

    function CubicSplitting(; separation_parameter=0.5, smoothing_ratio=0.5,
                            center_particle=true)
        new{typeof(separation_parameter)}(separation_parameter, smoothing_ratio,
                                          center_particle)
    end
end

@inline nchilds(system, refinement_pattern::CubicSplitting) = (2^ndims(system) +
                                                               refinement_pattern.center_particle)

function relative_position_children(system::System{2},
                                    refinement_pattern::CubicSplitting)
    (; smoothing_length) = system
    (; separation_parameter) = refinement_pattern

    direction_1 = [1 / sqrt(2), 1 / sqrt(2)]
    direction_2 = [1 / sqrt(2), -1 / sqrt(2)]
    direction_3 = -direction_1
    direction_4 = -direction_2

    # Is it `particle_spacing * separation_parameter` or `smoothing_length * separation_parameter`?
    relative_position = hcat(smoothing_length * separation_parameter * direction_1,
                             smoothing_length * separation_parameter * direction_2,
                             smoothing_length * separation_parameter * direction_3,
                             smoothing_length * separation_parameter * direction_4)

    if refinement_pattern.center_particle
        relative_position = hcat(relative_position, [0.0, 0.0])
    end

    return reinterpret(reshape, SVector{2, eltype(system)}, relative_position)
end

struct TriangularSplitting{ELTYPE}
    separation_parameter :: ELTYPE
    smoothing_ratio      :: ELTYPE
    center_particle      :: Bool

    function TriangularSplitting(; separation_parameter=0.5, smoothing_ratio=0.5,
                                 center_particle=true)
        new{typeof(separation_parameter)}(separation_parameter, smoothing_ratio,
                                          center_particle)
    end
end

@inline nchilds(system::System{2}, refinement_pattern::TriangularSplitting) = 3 +
                                                                              refinement_pattern.center_particle

function relative_position_children(system::System{2},
                                    refinement_pattern::TriangularSplitting)
    (; smoothing_length) = system
    (; separation_parameter) = refinement_pattern

    direction_1 = [0.0, 1.0]
    direction_2 = [-sqrt(3) / 2, -0.5]
    direction_3 = [sqrt(3) / 2, -0.5]

    relative_position = hcat(smoothing_length * separation_parameter * direction_1,
                             smoothing_length * separation_parameter * direction_2,
                             smoothing_length * separation_parameter * direction_3)

    if refinement_pattern.center_particle
        relative_position = hcat(relative_position, [0.0, 0.0])
    end

    return reinterpret(reshape, SVector{2, eltype(system)}, relative_position)
end

struct HexagonalSplitting{ELTYPE}
    separation_parameter :: ELTYPE
    smoothing_ratio      :: ELTYPE
    center_particle      :: Bool

    function HexagonalSplitting(; separation_parameter=0.5, smoothing_ratio=0.5,
                                center_particle=true)
        new{typeof(separation_parameter)}(separation_parameter, smoothing_ratio,
                                          center_particle)
    end
end

@inline nchilds(system::System{2}, refinement_pattern::HexagonalSplitting) = 6 +
                                                                             refinement_pattern.center_particle

function relative_position_children(system::System{2},
                                    refinement_pattern::HexagonalSplitting)
    (; smoothing_length) = system
    (; separation_parameter) = refinement_pattern

    direction_1 = [1.0, 0.0]
    direction_2 = [-1.0, 0.0]
    direction_3 = [0.5, sqrt(3) / 2]
    direction_4 = [0.5, -sqrt(3) / 2]
    direction_5 = [-0.5, sqrt(3) / 2]
    direction_6 = [-0.5, -sqrt(3) / 2]

    relative_position = hcat(smoothing_length * separation_parameter * direction_1,
                             smoothing_length * separation_parameter * direction_2,
                             smoothing_length * separation_parameter * direction_3,
                             smoothing_length * separation_parameter * direction_4,
                             smoothing_length * separation_parameter * direction_5,
                             smoothing_length * separation_parameter * direction_6)

    if refinement_pattern.center_particle
        relative_position = hcat(relative_position, [0.0, 0.0])
    end

    return reinterpret(reshape, SVector{2, eltype(system)}, relative_position)
end

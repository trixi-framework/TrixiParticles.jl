struct CubicSplitting{ELTYPE}
    epsilon           :: ELTYPE
    alpha             :: ELTYPE
    center_particle   :: Bool
    relative_position :: Vector{SVector{2, ELTYPE}}
    n_children        :: Int

    function CubicSplitting(; epsilon=0.5, alpha=0.5, center_particle=true)
        ELTYPE = typeof(epsilon)

        direction_1 = [1 / sqrt(2), 1 / sqrt(2)]
        direction_2 = [1 / sqrt(2), -1 / sqrt(2)]
        direction_3 = -direction_1
        direction_4 = -direction_2

        relative_position = [SVector{2, ELTYPE}(epsilon * direction_1),
            SVector{2, ELTYPE}(epsilon * direction_2),
            SVector{2, ELTYPE}(epsilon * direction_3),
            SVector{2, ELTYPE}(epsilon * direction_4)]

        if center_particle
            push!(relative_position, SVector(zero(ELTYPE), zero(ELTYPE)))
        end

        new{ELTYPE}(epsilon, alpha, center_particle, relative_position, 4 + center_particle)
    end
end

struct TriangularSplitting{ELTYPE}
    epsilon           :: ELTYPE
    alpha             :: ELTYPE
    center_particle   :: Bool
    relative_position :: Vector{SVector{2, ELTYPE}}
    n_children        :: Int

    function TriangularSplitting(; epsilon=0.5, alpha=0.5, center_particle=true)
        ELTYPE = typeof(epsilon)
        direction_1 = [0.0, 1.0]
        direction_2 = [-sqrt(3) / 2, -0.5]
        direction_3 = [sqrt(3) / 2, -0.5]

        relative_position = [SVector{2, ELTYPE}(epsilon * direction_1),
            SVector{2, ELTYPE}(epsilon * direction_2),
            SVector{2, ELTYPE}(epsilon * direction_3)]

        if center_particle
            push!(relative_position, SVector(zero(ELTYPE), zero(ELTYPE)))
        end

        new{ELTYPE}(epsilon, alpha, center_particle, relative_position, 3 + center_particle)
    end
end

struct HexagonalSplitting{ELTYPE}
    epsilon           :: ELTYPE
    alpha             :: ELTYPE
    center_particle   :: Bool
    relative_position :: Vector{SVector{2, ELTYPE}}
    n_children        :: Int

    function HexagonalSplitting(; epsilon=0.4, alpha=0.9, center_particle=true)
        ELTYPE = typeof(epsilon)

        direction_1 = [1.0, 0.0]
        direction_2 = [-1.0, 0.0]
        direction_3 = [0.5, sqrt(3) / 2]
        direction_4 = [0.5, -sqrt(3) / 2]
        direction_5 = [-0.5, sqrt(3) / 2]
        direction_6 = [-0.5, -sqrt(3) / 2]

        relative_position = [SVector{2, ELTYPE}(epsilon * direction_1),
            SVector{2, ELTYPE}(epsilon * direction_2),
            SVector{2, ELTYPE}(epsilon * direction_3),
            SVector{2, ELTYPE}(epsilon * direction_4),
            SVector{2, ELTYPE}(epsilon * direction_5),
            SVector{2, ELTYPE}(epsilon * direction_6)]

        if center_particle
            push!(relative_position, SVector(zero(ELTYPE), zero(ELTYPE)))
        end

        new{ELTYPE}(epsilon, alpha, center_particle, relative_position, 6 + center_particle)
    end
end

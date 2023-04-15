struct MergeShapes{ELTYPE <: Real}
    coordinates :: Array{ELTYPE, 2}
    velocities  :: Array{ELTYPE, 2}
    masses      :: Vector{ELTYPE}
    densities   :: Vector{ELTYPE}

    function MergeShapes(shapes...)
        ELTYPE = eltype(shapes[1].coordinates)

        coordinates = Tuple(shape.coordinates for shape in shapes)
        velocities = Tuple(shape.velocities for shape in shapes)
        masses = Tuple(shape.masses for shape in shapes)
        densities = Tuple(shape.densities for shape in shapes)

        coordinates = hcat(coordinates...)
        velocities = hcat(velocities...)
        masses = vcat(masses...)
        densities = vcat(densities...)

        return new{ELTYPE}(coordinates, velocities, masses, densities)
    end
end

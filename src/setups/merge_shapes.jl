"""
    MergeShapes(shapes...)

Merge multiple shapes into a single shape.

# Example
```julia
shape = MergeShapes(shape_1, shape_2, shape_3)
```
"""
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

        if overlapping_particles(coordinates)
            @warn "There are overlapping particles."
        end

        return new{ELTYPE}(coordinates, velocities, masses, densities)
    end
end

# Check overlapping particles.
function overlapping_particles(coords)

    # Round coordinates to make sure that particle position is unique
    coords = round.(coords, digits=6)

    tuple_coords = [Tuple(coords[:, i]) for i in axes(coords, 2)]

    return !allunique(tuple_coords)
end

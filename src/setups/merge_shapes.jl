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

        coordinates = hcat((shape.coordinates for shape in shapes)...)
        velocities = hcat((shape.velocities for shape in shapes)...)
        masses = vcat((shape.masses for shape in shapes)...)
        densities = vcat((shape.densities for shape in shapes)...)

        # TODO: Throw warning when overlapping particles detected

        return new{ELTYPE}(coordinates, velocities, masses, densities)
    end
end

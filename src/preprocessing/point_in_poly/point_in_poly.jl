function svector_points(points, ::Val{NDIMS}) where {NDIMS}
    if points isa AbstractMatrix
        if size(points, 1) != NDIMS
            throw(ArgumentError("point matrix must have $NDIMS rows"))
        end

        return reinterpret(reshape, SVector{NDIMS, eltype(points)}, points)
    end

    return points
end

include("hierarchical_winding.jl")
include("winding_number_hormann.jl")
include("winding_number_jacobson.jl")

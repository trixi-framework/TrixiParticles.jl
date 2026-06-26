function wrap_points(points, ::Val{NDIMS}) where {NDIMS}
    if points isa AbstractMatrix
        if size(points, 1) != NDIMS
            throw(ArgumentError("point matrix must have $NDIMS rows"))
        end

        # Interpret an `NDIMS`-by-`N` matrix as one static vector per column.
        return reinterpret(reshape, SVector{NDIMS, eltype(points)}, points)
    end

    return points
end

include("geometries/geometries.jl")
include("point_in_poly/point_in_poly.jl")
include("particle_packing/particle_packing.jl")

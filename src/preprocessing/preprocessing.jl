function wrap_points(points, ::Val{NDIMS}) where {NDIMS}
    if points isa AbstractMatrix
        if size(points, 1) != NDIMS
            throw(ArgumentError("point matrix must have $NDIMS rows"))
        end

        # Interpret an `NDIMS`-by-`N` matrix as one static vector per column. Constructing
        # the vectors explicitly also supports non-contiguous matrix views.
        return map(eachcol(points)) do point
            return SVector{NDIMS, eltype(points)}(point)
        end
    end

    return points
end

include("geometries/geometries.jl")
include("point_in_poly/point_in_poly.jl")
include("particle_packing/particle_packing.jl")

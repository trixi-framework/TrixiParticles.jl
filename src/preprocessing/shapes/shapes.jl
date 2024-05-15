abstract type Shapes{NDIMS} end

include("polygon_shape.jl")
include("triangle_mesh.jl")
include("io.jl")

@inline Base.ndims(::Shapes{NDIMS}) where {NDIMS} = NDIMS

@inline Base.eltype(shape::Shapes) = eltype(first(first(shape.min_box)))

@inline eachface(mesh) = Base.OneTo(nfaces(mesh))

@inline point_position(A, shape, i) = extract_svector(A, Val(ndims(shape)), i)

include("polygon_shape.jl")
include("triangle_mesh.jl")
include("io.jl")

@inline eachface(mesh) = Base.OneTo(nfaces(mesh))

@inline point_position(A, shape, i) = extract_svector(A, Val(ndims(shape)), i)

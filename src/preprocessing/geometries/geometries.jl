include("polygon.jl")
include("triangle_mesh.jl")
include("io.jl")

@inline eachface(mesh) = Base.OneTo(nfaces(mesh))

@inline point_position(A, geometry, i) = extract_svector(A, Val(ndims(geometry)), i)

include("polygon.jl")
include("triangle_mesh.jl")
include("io.jl")

@inline eachface(mesh) = Base.OneTo(nfaces(mesh))

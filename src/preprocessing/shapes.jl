abstract type Shapes{NDIMS} end

include("polygon_shape.jl")
include("triangle_mesh.jl")
include("point_in_poly/algorithm.jl")

@inline Base.ndims(::Shapes{NDIMS}) where {NDIMS} = NDIMS

# Note: `n_vertices`-1, since the last vertex is the same as the first one
@inline eachvertices(shape) = Base.OneTo(shape.n_vertices - 1)
@inline eachfaces(shape::TriangleMesh) = Base.OneTo(shape.n_faces)
@inline eachfaces(shape::Polygon) = eachvertices(shape)

@inline min_corner(vertices, dim, pad) = minimum(vertices[dim, :]) - pad
@inline max_corner(vertices, dim, pad) = maximum(vertices[dim, :]) + pad

@inline function position(A, shape, i)
    return TrixiParticles.extract_svector(A, Val(ndims(shape)), i)
end

@inline function position(A, ::Val{2}, i)
    return TrixiParticles.extract_svector(A, Val(2), i)
end

function particle_grid(vertices, particle_spacing; pad=2*particle_spacing)
    NDIMS = size(vertices, 1)

    function ranges(dim)
        min_corner(vertices, dim, pad):particle_spacing:max_corner(vertices, dim, pad)
    end

    ranges_ = ntuple(dim -> ranges(dim), NDIMS)

    return hcat(collect.(Iterators.product(ranges_...))...)
end

function sample(shape, particle_spacing; point_in_poly=WindingNumberHorman(shape))
    grid = particle_grid(shape.vertices, particle_spacing)

    inpoly = point_in_poly(shape, grid)
    coordinates = grid[:, inpoly]

    return coordinates #InitialCondition(coordinates, velocities, masses, densities)
end

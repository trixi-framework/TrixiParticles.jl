using TrixiParticles
using FileIO
using LinearAlgebra: cross, dot, normalize
using SimpleUnPack: @unpack

struct WindingNumberHorman{}
    quadrant_numbers::Array{Int, 1}

    function WindingNumberHorman(shape)
        quadrant_numbers = zeros(Int, shape.n_vertices)

        return new{}(quadrant_numbers)
    end
end

struct Polygon{ELTYPE}
    vertices   :: Array{ELTYPE, 2}
    edges      :: NTuple
    n_vertices :: Int

    function Polygon(vertices)
        if !(vertices[:, end] ≈ vertices[:, 1])
            error("The first and last vertex of the polygon must be the same.")
        end
        n_vertices = size(vertices, 2)
        ELTYPE = eltype(vertices)

        edges = (;)
        for i in n_vertices - 1
            v1 = vertices[:, i]
            v2 = vertices[:, i + 1]
            edges = (edges..., (v1, v2))
        end

        return new{ELTYPE}(vertices, edges, n_vertices)
    end
end

struct Polyhedron{NDIMS, ELTYPE}
    vertices   :: Array{ELTYPE, 2}
    normals    :: Array{ELTYPE, 2}
    faces      :: NTuple
    n_vertices :: Int

    function Polyhedron(mesh)
        ELTYPE = eltype(first(mesh.position))
        NDIMS = length(first(mesh))
        NELEMENTS = length(mesh)

        normals = zeros(NDIMS, NELEMENTS)

        faces = (;)
        face_id = 0
        for face in mesh
            face_id += 1

            v1, v2, v3 = face
            a = v2 - v1
            b = v3 - v1
            normals[:, face_id] .= normalize(cross(a, b))
            faces = (faces..., (v1, v2, v3))
        end

        vertices = stack(union(mesh.position))
        n_vertices = size(vertices, 2)

        return new{NDIMS, ELTYPE}(vertices, normals, faces, n_vertices)
    end
end

@inline min_corner(vertices, dim, pad) = minimum(vertices[dim, :]) - pad
@inline max_corner(vertices, dim, pad) = maximum(vertices[dim, :]) + pad

## wrap function
@inline function position(A, i)
    return TrixiParticles.extract_svector(A, Val(2), i)
end

function particle_grid(vertices, particle_spacing)
    NDIMS = size(vertices, 1)
    pad = 2 * particle_spacing

    function ranges(dim)
        min_corner(vertices, dim, pad):particle_spacing:max_corner(vertices, dim, pad)
    end

    ranges_ = ntuple(dim -> ranges(dim), NDIMS)

    return hcat(collect.(Iterators.product(ranges_...))...)
end

function sample(shape, particle_spacing; point_in_poly=WindingNumberHorman(shape))
    grid = particle_grid(shape.vertices, particle_spacing)

    coordinates = grid[:, point_in_poly(shape, grid)]

    return coordinates#InitialCondition(coordinates, velocities, masses, densities)
end

function (point_in_poly::WindingNumberHorman)(shape, points)
    @unpack quadrant_numbers = point_in_poly
    @unpack vertices = shape

    inpoly = trues(size(points, 2))

    h_unit = SVector(1.0, 0.0)
    v_unit = SVector(0.0, 1.0)

    function quadrant(dot_v, dot_h)
        (dot_v >= 0.0 && dot_h > 0.0) && return 0
        (dot_v > 0.0 && dot_h <= 0.0) && return 1
        (dot_v <= 0.0 && dot_h < 0.0) && return 2
        (dot_v < 0.0 && dot_h >= 0.0) && return 3
    end

    for querry_point in axes(points, 2)
        for vertex in eachvertices(shape)
            direction = position(vertices, vertex) - position(points, querry_point)
            dot_v = dot(direction, v_unit)
            dot_h = dot(direction, h_unit)

            quadrant_numbers[vertex] = quadrant(dot_v, dot_h)
        end

        # the last vertex is the same as the first one.
        quadrant_numbers[end] = quadrant_numbers[1]

        winding_number = 0
        for vertex in eachvertices(shape)
            v1 = position(vertices, vertex)
            v2 = position(vertices, vertex + 1)
            v_query = position(points, querry_point)

            # because 0 <= `quadrant_numbers` <= 3 we know that -3 <= `quarter_angel` <= 3
            quarter_angel = quadrant_numbers[vertex + 1] - quadrant_numbers[vertex]
            positiv_det = positive_determinant(v1, v2, v_query)

            if quarter_angel == -3
                winding_number += 1
            elseif quarter_angel == 3
                winding_number -= 1
            elseif quarter_angel == -2 && positiv_det
                winding_number += 1
            elseif quarter_angel == 2 && !positiv_det
                winding_number -= 1
            end
        end

        (winding_number == 0) && (inpoly[querry_point] = false)
    end

    return inpoly
end

function positive_determinant(v1, v2, v_query)

    # To check  the orientation of the triangle Δ(`v_query`, `v1`, `v2`) by finding the
    # sign of the determinant.
    # `+`: counter clockwise
    # `-`: clockwise
    positive_sign = (v1[1] - v_query[1]) * (v2[2] - v_query[2]) -
                    (v2[1] - v_query[1]) * (v1[2] - v_query[2]) > 0.0

    return positive_sign
end

# Note: `n_vertices`-1, since the last vertex is the same as the first one
@inline eachvertices(shape) = Base.OneTo(shape.n_vertices - 1)

dir = joinpath("Data", "stl-files", "examples")
mesh = load(joinpath(expanduser("~/") * dir, "bar.stl"))

polygon = Polygon([1.8 2.4 6.2 8.5 5.1 3.4 1.8;
                   4.2 1.7 0.8 3.2 5.1 0.2 4.2])

polyhedron = Polyhedron(mesh)

coords = sample(polygon, 0.1)
trixi2vtk(coords, filename="test_coords")
trixi2vtk(polygon.vertices, filename="polygon_coords")

struct TriangleMesh{NDIMS, ELTYPE, F} <: Shapes{NDIMS}
    vertices   :: Array{ELTYPE, 2}
    normals    :: Array{ELTYPE, 2}
    faces      :: F
    n_vertices :: Int
    n_faces    :: Int

    function TriangleMesh(mesh)
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

        return new{NDIMS, ELTYPE, typeof(faces)}(vertices, normals, faces, n_vertices,
                                                 NELEMENTS)
    end
end

struct WindingNumberJacobson end

function (point_in_poly::WindingNumberJacobson)(mesh, points)
    faces = unpack_faces_or_edges(mesh)
    inpoly = falses(size(points, 2))

    @threaded for querry_point in axes(points, 2)
        p = position(points, mesh, querry_point)

        winding_number = 0.0
        for face in faces
            signed_angle = inverse_tangent_function(face, p)
            winding_number += signed_angle
        end

        winding_number /= pi_factor(Val(ndims(mesh)))

        # TODO make this condition better (`winding_number` != 0)
        if !(-sqrt(eps()) < winding_number < sqrt(eps()))
            inpoly[querry_point] = true
        end
    end

    return inpoly
end

# A. Van Oosterom 1983,
# The Solid Angle of a Plane Triangle (doi: 10.1109/TBME.1983.325207)
function inverse_tangent_function(face, p::SVector{3})
    a = face[1] - p
    b = face[2] - p
    c = face[3] - p
    a_ = norm(a)
    b_ = norm(b)
    c_ = norm(c)

    divisor = a_ * b_ * c_ + dot(a, b) * c_ + dot(b, c) * a_ + dot(c, a) * b_

    return 2 * atan(det([a b c]) , divisor)
end

function inverse_tangent_function(face, p::SVector{2})
    a = face[1] - p
    b = face[2] - p

    return atan(det([a b]) , (dot(a, b))) #acos(cos_theta)
end

@inline unpack_faces_or_edges(mesh::Polygon) = mesh.edges
@inline unpack_faces_or_edges(mesh::TriangleMesh) = mesh.faces

@inline pi_factor(::Val{3}) = 4pi
@inline pi_factor(::Val{2}) = 2pi

struct WindingNumberJacobson end

function (point_in_poly::WindingNumberJacobson)(mesh, points)
    faces = unpack_faces_or_edges(mesh)
    inpoly = falses(size(points, 2))

    for querry_point in axes(points, 2)
        p = position(points, mesh, querry_point)

        winding_number = 0.0
        for face in eachfaces(mesh)
            signed_angle = inverse_tangent_function(faces, face, p, mesh)
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
function inverse_tangent_function(faces, face_id, p::SVector{3}, shape)
    v1 = position(view(faces, :, :, face_id), shape, 1)
    v2 = position(view(faces, :, :, face_id), shape, 2)
    v3 = position(view(faces, :, :, face_id), shape, 3)
    a = v1 - p
    b = v2 - p
    c = v3 - p
    a_ = norm(a)
    b_ = norm(b)
    c_ = norm(c)

    divisor = a_ * b_ * c_ + dot(a, b) * c_ + dot(b, c) * a_ + dot(c, a) * b_

    return 2 * atan(det([a b c]), divisor)
end

function inverse_tangent_function(edges, edge, p::SVector{2}, shape)
    v1 = position(view(edges, :, :, edge), shape, 1)
    v2 = position(view(edges, :, :, edge), shape, 2)
    a = v1 - p
    b = v2 - p

    return atan(det([a b]), (dot(a, b)))
end

@inline unpack_faces_or_edges(mesh::Polygon) = mesh.edges
@inline unpack_faces_or_edges(mesh::TriangleMesh) = mesh.faces

@inline pi_factor(::Val{3}) = 4pi
@inline pi_factor(::Val{2}) = 2pi

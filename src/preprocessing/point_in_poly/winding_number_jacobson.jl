struct NaiveWinding end

struct HierarchicalWinding{BB}
    bounding_box::BB
    function HierarchicalWinding(bounding_box)
        return new{typeof(bounding_box)}(bounding_box)
    end
end

# Alec Jacobson, Ladislav Kavan, and Olga Sorkine-Hornung. 2013.
# Robust inside-outside segmentation using generalized winding numbers.
# ACM Trans. Graph. 32, 4, Article 33 (July 2013), 12 pages.
# https://doi.org/10.1145/2461912.2461916

struct WindingNumberJacobson{ELTYPE}
    winding_number_factor :: ELTYPE
    winding               :: Union{NaiveWinding, HierarchicalWinding}

    function WindingNumberJacobson(; winding_number_factor=sqrt(eps()),
                                   winding=NaiveWinding())
        ELTYPE = typeof(winding_number_factor)
        return new{ELTYPE}(winding_number_factor, winding)
    end
end

function (point_in_poly::WindingNumberJacobson)(mesh::Shapes{3}, points)
    (; winding_number_factor, winding) = point_in_poly

    inpoly = falses(size(points, 2))

    @threaded for query_point in axes(points, 2)
        p = point_position(points, mesh, query_point)

        winding_number = winding(mesh, p)

        winding_number /= 4pi

        # `(winding_number != 0.0)`
        if !(-winding_number_factor < winding_number < winding_number_factor)
            inpoly[query_point] = true
        end
    end

    return inpoly
end

function (point_in_poly::WindingNumberJacobson)(mesh::Shapes{2}, points)
    (; winding_number_factor) = point_in_poly
    (; edge_vertices) = mesh
    inpoly = falses(size(points, 2))

    @threaded for query_point in axes(points, 2)
        p = point_position(points, mesh, query_point)

        winding_number = sum(edge_vertices) do edge
            a = edge[1] - p
            b = edge[2] - p

            return atan(det([a b]), (dot(a, b)))
        end

        winding_number /= 2pi

        # `(winding_number != 0.0)`
        if !(-winding_number_factor < winding_number < winding_number_factor)
            inpoly[query_point] = true
        end
    end

    return inpoly
end

@inline function (winding::NaiveWinding)(mesh, query_point)
    (; face_vertices) = mesh

    return naive_winding(mesh, face_vertices, query_point)
end

@inline function (winding::HierarchicalWinding)(mesh, query_point)
    (; bounding_box) = winding

    return hierarchical_winding(bounding_box, mesh, query_point)
end

@inline function naive_winding(mesh, faces, query_point)
    winding_number = sum(faces, init=0.0) do face

        # A. Van Oosterom 1983,
        # The Solid Angle of a Plane Triangle (doi: 10.1109/TBME.1983.325207)
        a = face_vertex(mesh, face, 1) - query_point
        b = face_vertex(mesh, face, 2) - query_point
        c = face_vertex(mesh, face, 3) - query_point
        a_ = norm(a)
        b_ = norm(b)
        c_ = norm(c)

        divisor = a_ * b_ * c_ + dot(a, b) * c_ + dot(b, c) * a_ + dot(c, a) * b_

        return 2atan(det([a b c]), divisor)
    end

    return winding_number
end

# `face` holds the coordinates of each vertex
@inline face_vertex(mesh, face, index) = face[index]

# `face` holds the index of each vertex
@inline function face_vertex(mesh, face::NTuple{3, Int}, index)
    v_id = face[index]

    return mesh.vertices[v_id]
end

# `face` is the index of the face
@inline function face_vertex(mesh, face::Int, index)
    (; face_vertices) = mesh

    return face_vertices[face][index]
end

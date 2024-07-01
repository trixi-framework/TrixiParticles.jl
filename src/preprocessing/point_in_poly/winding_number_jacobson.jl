struct NaiveWinding end

struct HierarchicalWinding{BB}
    bounding_box::BB

    function HierarchicalWinding(shape)
        bounding_box = BoundingBoxTree(eachface(shape), shape.min_corner, shape.max_corner)

        directed_edges = zeros(Int, length(shape.normals_edge))

        construct_hierarchy!(bounding_box, shape, directed_edges)

        return new{typeof(bounding_box)}(bounding_box)
    end
end

"""
    WindingNumberJacobson(; shape=nothing, winding_number_factor=sqrt(eps()),
                          hierarchical_winding=false)
Algorithm for inside-outside segmentation of a complex shape proposed by Jacobson et al. (2013).

# Keywords
- `shape`: Complex shape returned by [`load_shape`](@ref) and is only required when using
           `hierarchical_winding=true`.
- `hierarchical_winding`: If set to `true`, an optimised hierarchical approach will be used
                          which gives a significant speedup.
                          It is only supported for 3D shapes yet.
- `winding_number_factor`: For leaky shapes, a factor of `0.4` will give a better inside-outside segmentation.
"""
struct WindingNumberJacobson{ELTYPE}
    winding_number_factor :: ELTYPE
    winding               :: Union{NaiveWinding, HierarchicalWinding}

    function WindingNumberJacobson(; shape=nothing, winding_number_factor=sqrt(eps()),
                                   hierarchical_winding=false)
        if hierarchical_winding && shape isa Nothing
            throw(ArgumentError("`shape` must be of type `Shapes` when using hierarchical winding"))
        end

        winding = hierarchical_winding ? HierarchicalWinding(shape) : NaiveWinding()

        return new{typeof(winding_number_factor)}(winding_number_factor, winding)
    end
end

function (point_in_poly::WindingNumberJacobson)(mesh::Shapes{3}, points;
                                                store_winding_number=false)
    (; winding_number_factor, winding) = point_in_poly

    inpoly = falses(size(points, 2))

    winding_numbers = Float64[]
    store_winding_number && (winding_numbers = resize!(winding_numbers, length(inpoly)))

    @threaded points for query_point in axes(points, 2)
        p = point_position(points, mesh, query_point)

        winding_number = winding(mesh, p)

        winding_number /= 4pi

        store_winding_number && (winding_numbers[query_point] = winding_number)

        # Relaxed restriction of `(winding_number != 0.0)`
        if !(-winding_number_factor < winding_number < winding_number_factor)
            inpoly[query_point] = true
        end
    end

    return inpoly, winding_numbers
end

function (point_in_poly::WindingNumberJacobson)(mesh::Shapes{2}, points;
                                                store_winding_number=false)
    (; winding_number_factor) = point_in_poly
    (; edge_vertices) = mesh

    inpoly = falses(size(points, 2))

    winding_numbers = Float64[]
    store_winding_number && (winding_numbers = resize!(winding_numbers, length(inpoly)))

    @threaded points for query_point in axes(points, 2)
        p = point_position(points, mesh, query_point)

        winding_number = sum(edge_vertices, init=0.0) do edge
            a = edge[1] - p
            b = edge[2] - p

            return atan(det([a b]), (dot(a, b)))
        end

        winding_number /= 2pi

        store_winding_number && (winding_numbers[query_point] = winding_number)

        # Relaxed restriction of `(winding_number != 0.0)`
        if !(-winding_number_factor < winding_number < winding_number_factor)
            inpoly[query_point] = true
        end
    end

    return inpoly, winding_numbers
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

# The following functions distinguish between actual triangles and reconstructed triangles
# in the hierarchical winding approach

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

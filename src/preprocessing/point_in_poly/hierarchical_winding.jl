# This bounding box is used for the hierarchical evaluation of the `WindingNumberJacobsen`.
# It is implementing a binary tree and thus stores the left and right child and also the
# faces and closing faces which are inside the bounding box.
struct BoundingBoxTree{MC}
    faces         :: Vector{Int}
    min_corner    :: MC
    max_corner    :: MC
    is_leaf       :: Bool
    closing_faces :: Vector{NTuple{3, Int}}
    child_left    :: BoundingBoxTree
    child_right   :: BoundingBoxTree

    function BoundingBoxTree(geometry, faces, directed_edges, min_corner, max_corner)
        closing_faces = Vector{NTuple{3, Int}}()

        if length(faces) < 100
            return new{typeof(min_corner)}(faces, min_corner, max_corner, true,
                                           closing_faces)
        end

        determine_closure!(closing_faces, min_corner, max_corner, geometry, faces,
                           directed_edges)

        if length(closing_faces) >= length(faces)
            return new{typeof(min_corner)}(faces, min_corner, max_corner, true,
                                           closing_faces)
        end

        # Bisect the box splitting its longest side
        box_edges = max_corner - min_corner

        split_direction = argmax(box_edges)

        uvec = (1:3) .== split_direction

        max_corner_left = max_corner - 0.5box_edges[split_direction] * uvec
        min_corner_right = min_corner + 0.5box_edges[split_direction] * uvec

        faces_left = is_in_box(geometry, faces, min_corner, max_corner_left)
        faces_right = is_in_box(geometry, faces, min_corner_right, max_corner)

        child_left = BoundingBoxTree(geometry, faces_left, directed_edges,
                                     min_corner, max_corner_left)
        child_right = BoundingBoxTree(geometry, faces_right, directed_edges,
                                      min_corner_right, max_corner)

        return new{typeof(min_corner)}(faces, min_corner, max_corner, false, closing_faces,
                                       child_left, child_right)
    end
end

struct HierarchicalWinding{BB}
    bounding_box::BB

    function HierarchicalWinding(geometry)
        min_corner = geometry.min_corner .- sqrt(eps())
        max_corner = geometry.max_corner .+ sqrt(eps())

        directed_edges = zeros(Int, length(geometry.edge_normals))

        bounding_box = BoundingBoxTree(geometry, eachface(geometry), directed_edges,
                                       min_corner, max_corner)

        return new{typeof(bounding_box)}(bounding_box)
    end
end

@inline function (winding::HierarchicalWinding)(mesh, query_point)
    (; bounding_box) = winding

    return hierarchical_winding(bounding_box, mesh, query_point)
end

function hierarchical_winding(bounding_box, mesh, query_point)
    (; min_corner, max_corner) = bounding_box

    if bounding_box.is_leaf
        return naive_winding(mesh, bounding_box.faces, query_point)

    elseif !is_in_box(query_point, min_corner, max_corner)
        # `query_point` is outside bounding box
        return -naive_winding(mesh, bounding_box.closing_faces, query_point)
    end

    winding_number_left = hierarchical_winding(bounding_box.child_left, mesh, query_point)
    winding_number_right = hierarchical_winding(bounding_box.child_right, mesh, query_point)

    return winding_number_left + winding_number_right
end

# This only works when all `vertices` are unique
function determine_closure!(closing_faces, min_corner, max_corner, mesh::TriangleMesh{3},
                            faces, directed_edges)
    (; edge_vertices_ids, face_vertices_ids, face_edges_ids, vertices) = mesh

    directed_edges .= 0
    intersecting_faces = Int[]

    # Find all exterior edges
    for face in faces
        v1 = face_vertices_ids[face][1]
        v2 = face_vertices_ids[face][2]
        v3 = face_vertices_ids[face][3]

        if is_in_box(vertices[v1], min_corner, max_corner) &&
           is_in_box(vertices[v2], min_corner, max_corner) &&
           is_in_box(vertices[v3], min_corner, max_corner)
            # Face is completely inside the bounding box

            edge1 = face_edges_ids[face][1]
            edge2 = face_edges_ids[face][2]
            edge3 = face_edges_ids[face][3]

            if edge_vertices_ids[edge1] == (v1, v2)
                directed_edges[edge1] += 1
            else
                directed_edges[edge1] -= 1
            end
            if edge_vertices_ids[edge2] == (v2, v3)
                directed_edges[edge2] += 1
            else
                directed_edges[edge2] -= 1
            end
            if edge_vertices_ids[edge3] == (v3, v1)
                directed_edges[edge3] += 1
            else
                directed_edges[edge3] -= 1
            end

        else # Face is intersecting the boundaries
            push!(intersecting_faces, face)
        end
    end

    exterior_edges = findall(!iszero, directed_edges)
    resize!(closing_faces, 0)

    if !isempty(exterior_edges)
        closing_vertex = edge_vertices_ids[exterior_edges[1]][2]
    end

    for edge in exterior_edges
        v1 = edge_vertices_ids[edge][1]
        v2 = edge_vertices_ids[edge][2]

        if v1 == closing_vertex || v2 == closing_vertex
            continue
        end

        if directed_edges[edge] < 0
            for _ in 1:abs(directed_edges[edge])
                push!(closing_faces, (v1, v2, closing_vertex))
            end
        elseif directed_edges[edge] > 0
            for _ in 1:abs(directed_edges[edge])
                push!(closing_faces, (v2, v1, closing_vertex))
            end
        end
    end

    for face in intersecting_faces
        v1 = face_vertices_ids[face][1]
        v2 = face_vertices_ids[face][2]
        v3 = face_vertices_ids[face][3]

        # Flip order of vertices
        push!(closing_faces, (v3, v2, v1))
    end

    return closing_faces
end

function is_in_box(mesh, faces, min_corner, max_corner)
    return filter(face -> is_in_box(barycenter(mesh, face), min_corner, max_corner), faces)
end

@inline function barycenter(mesh::Polygon{2}, edge)
    (; edge_vertices) = mesh

    v1 = edge_vertices[edge][1]
    v2 = edge_vertices[edge][2]

    return 0.5(v1 + v2)
end

@inline function barycenter(mesh::TriangleMesh{3}, face)
    (; face_vertices) = mesh

    v1 = face_vertices[face][1]
    v2 = face_vertices[face][2]
    v3 = face_vertices[face][3]

    return (v1 + v2 + v3) / 3
end

@inline function is_in_box(p::SVector{2}, min_corner, max_corner)
    p[1] < min_corner[1] && return false
    p[1] >= max_corner[1] && return false

    p[2] < min_corner[2] && return false
    p[2] >= max_corner[2] && return false

    return true
end

@inline function is_in_box(p::SVector{3}, min_corner, max_corner)
    p[1] < min_corner[1] && return false
    p[1] >= max_corner[1] && return false

    p[2] < min_corner[2] && return false
    p[2] >= max_corner[2] && return false

    p[3] < min_corner[3] && return false
    p[3] >= max_corner[3] && return false

    return true
end

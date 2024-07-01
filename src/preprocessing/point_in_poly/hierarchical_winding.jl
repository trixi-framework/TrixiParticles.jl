struct BoundingBoxTree{MC}
    faces         :: Vector{Int}
    min_corner    :: MC
    max_corner    :: MC
    is_leaf       :: Ref{Bool}
    closing_faces :: Vector{NTuple{3, Int}}
    children      :: Vector{BoundingBoxTree}

    function BoundingBoxTree(faces, min_corner, max_corner)
        closing_faces = Vector{NTuple{3, Int}}()
        children = Vector{BoundingBoxTree}()

        return new{typeof(min_corner)}(faces, min_corner, max_corner, Ref(false),
                                       closing_faces, children)
    end
end

function hierarchical_winding(bounding_box, mesh, query_point)
    (; min_corner, max_corner) = bounding_box

    if bounding_box.is_leaf[]
        return naive_winding(mesh, bounding_box.faces, query_point)

    elseif !in_bounding_box(query_point, min_corner, max_corner)
        # `query_point` is outside bounding box
        return -naive_winding(mesh, bounding_box.closing_faces, query_point)
    end

    winding_number_left = hierarchical_winding(bounding_box.children[1], mesh, query_point)
    winding_number_right = hierarchical_winding(bounding_box.children[2], mesh, query_point)

    return winding_number_left + winding_number_right
end

function construct_hierarchy!(bounding_box, mesh, directed_edges)
    (; max_corner, min_corner, faces) = bounding_box

    if length(faces) < 100
        bounding_box.is_leaf[] = true

        return bounding_box
    end

    determine_closure!(bounding_box, mesh, faces, directed_edges)

    if length(bounding_box.closing_faces) >= length(faces)
        bounding_box.is_leaf[] = true

        return bounding_box
    end

    # Bisect the box splitting its longest side
    box_edges = max_corner - min_corner

    split_direction = findfirst(x -> maximum(box_edges) == x, box_edges)

    uvec = (1:3) .== split_direction

    max_corner_left = max_corner - 0.5box_edges[split_direction] * uvec
    min_corner_right = min_corner + 0.5box_edges[split_direction] * uvec

    faces_left = in_bbox(mesh, faces, min_corner, max_corner_left)
    faces_right = in_bbox(mesh, faces, min_corner_right, max_corner)

    bbox_left = BoundingBoxTree(faces_left, min_corner, max_corner_left)
    bbox_right = BoundingBoxTree(faces_right, min_corner_right, max_corner)

    push!(bounding_box.children, bbox_left)
    push!(bounding_box.children, bbox_right)

    construct_hierarchy!(bbox_left, mesh, directed_edges)
    construct_hierarchy!(bbox_right, mesh, directed_edges)

    return bounding_box
end

# This only works when all `vertices` are unique
function determine_closure!(bounding_box, mesh::Shapes{3}, faces, count_directed_edge)
    (; edge_vertices_ids, face_vertices_ids, face_edges_ids, vertices) = mesh
    (; min_corner, max_corner) = bounding_box

    count_directed_edge .= 0
    intersecting_faces = Int[]

    # Find all exterior edges
    for face in faces
        v1 = face_vertices_ids[face][1]
        v2 = face_vertices_ids[face][2]
        v3 = face_vertices_ids[face][3]

        if in_bounding_box(vertices[v1], min_corner, max_corner) &&
           in_bounding_box(vertices[v2], min_corner, max_corner) &&
           in_bounding_box(vertices[v3], min_corner, max_corner)
            # Face is completely inside the bounding box

            edge1 = face_edges_ids[face][1]
            edge2 = face_edges_ids[face][2]
            edge3 = face_edges_ids[face][3]

            if edge_vertices_ids[edge1] == (v1, v2)
                count_directed_edge[edge1] += 1
            else
                count_directed_edge[edge1] -= 1
            end
            if edge_vertices_ids[edge2] == (v2, v3)
                count_directed_edge[edge2] += 1
            else
                count_directed_edge[edge2] -= 1
            end
            if edge_vertices_ids[edge3] == (v3, v1)
                count_directed_edge[edge3] += 1
            else
                count_directed_edge[edge3] -= 1
            end

        elseif !in_bounding_box(vertices[v1], min_corner, max_corner) ||
               !in_bounding_box(vertices[v2], min_corner, max_corner) ||
               !in_bounding_box(vertices[v3], min_corner, max_corner)
            # Face is intersecting the boundaries

            push!(intersecting_faces, face)
        end
    end

    closing_edges = findall(!iszero, count_directed_edge)
    resize!(bounding_box.closing_faces, 0)

    if !isempty(closing_edges)
        closing_vertex = edge_vertices_ids[closing_edges[1]][2]
    end

    for edge in closing_edges
        v1 = edge_vertices_ids[edge][1]
        v2 = edge_vertices_ids[edge][2]

        if v1 == closing_vertex || v2 == closing_vertex
            continue
        end

        if count_directed_edge[edge] < 0
            @inbounds for _ in 1:abs(count_directed_edge[edge])
                push!(bounding_box.closing_faces, (v1, v2, closing_vertex))
            end
        elseif count_directed_edge[edge] > 0
            @inbounds for _ in 1:abs(count_directed_edge[edge])
                push!(bounding_box.closing_faces, (v2, v1, closing_vertex))
            end
        end
    end

    for face in intersecting_faces
        v1 = face_vertices_ids[face][1]
        v2 = face_vertices_ids[face][2]
        v3 = face_vertices_ids[face][3]

        # Flip order of vertices
        push!(bounding_box.closing_faces, (v3, v2, v1))
    end

    return bounding_box
end

function in_bbox(mesh, faces, min_corner, max_corner)
    faces_in_bbox = Int[]

    for face in faces
        if in_bounding_box(barycenter(mesh, face), min_corner, max_corner)
            push!(faces_in_bbox, face)
        end
    end

    return faces_in_bbox
end

@inline function barycenter(mesh::Shapes{2}, edge)
    (; edge_vertices) = mesh

    v1 = edge_vertices[edge][1]
    v2 = edge_vertices[edge][2]

    return 0.5(v1 + v2)
end

@inline function barycenter(mesh::Shapes{3}, face)
    (; face_vertices) = mesh

    v1 = face_vertices[face][1]
    v2 = face_vertices[face][2]
    v3 = face_vertices[face][3]

    return (v1 + v2 + v3) / 3
end

@inline function in_bounding_box(p::SVector{2}, min_corner, max_corner)
    p[1] < min_corner[1] && return false
    p[1] >= max_corner[1] && return false

    p[2] < min_corner[2] && return false
    p[2] >= max_corner[2] && return false

    return true
end

@inline function in_bounding_box(p::SVector{3}, min_corner, max_corner)
    p[1] < min_corner[1] && return false
    p[1] >= max_corner[1] && return false

    p[2] < min_corner[2] && return false
    p[2] >= max_corner[2] && return false

    p[3] < min_corner[3] && return false
    p[3] >= max_corner[3] && return false

    return true
end

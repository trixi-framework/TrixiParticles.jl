# This bounding box is used for the hierarchical evaluation of the `WindingNumberJacobsen`.
# It is implementing a binary tree and thus stores the left and right child and also the
# faces and closing faces which are inside the bounding box.
struct BoundingBoxTree{VOVOT, VOSV, V}
    depth         :: Int    # Depth of the tree
    faces         :: VOVOT  # VectorOfVectors{NTuple{NDIMS, Int}}()
    closing_faces :: VOVOT  # VectorOfVectors{NTuple{NDIMS, Int}}()
    min_corners   :: VOSV   # Vector{SVector{NDIMS}}()
    max_corners   :: VOSV   # Vector{SVector{NDIMS}}()
    childs_left   :: V      # Vector{Int}
    childs_right  :: V      # Vector{Int}
end

function BoundingBoxTree(geometry::Geometry)
    ELTYPE = eltype(geometry)

    # Note that overlapping bounding boxes are perfectly fine
    min_corner = geometry.min_corner .- ELTYPE(sqrt(eps()))
    max_corner = geometry.max_corner .+ ELTYPE(sqrt(eps()))

    if ndims(geometry) == 3
        directed_edges = zeros(Int, length(geometry.edge_normals))
    else
        directed_edges = zeros(Int, length(geometry.vertices))
    end

    faces = VectorOfVectors{NTuple{ndims(geometry), Int}}()
    closing_faces = VectorOfVectors{NTuple{ndims(geometry), Int}}()
    min_corners = Vector{SVector{ndims(geometry), ELTYPE}}()
    max_corners = Vector{SVector{ndims(geometry), ELTYPE}}()
    childs_left = Int[]
    childs_right = Int[]

    build_tree!(faces, closing_faces, min_corners, max_corners,
                childs_left, childs_right, directed_edges,
                geometry, eachface(geometry), min_corner, max_corner)

    return BoundingBoxTree(length(faces), faces, closing_faces, min_corners, max_corners,
                           childs_left, childs_right)
end

function build_tree!(faces, closing_faces, min_corners, max_corners,
                     childs_left, childs_right, directed_edges,
                     geometry, face_ids, min_corner_local, max_corner_local)
    NDIMS = ndims(geometry)

    closing_faces_local = NTuple{NDIMS, Int}[]

    max_faces_in_box = NDIMS == 3 ? 100 : 20
    if length(face_ids) < max_faces_in_box
        push!(faces, faces_to_tuple(face_ids, geometry))
        push!(min_corners, min_corner_local)
        push!(max_corners, max_corner_local)
        push!(childs_left, -1)
        push!(childs_right, -1)
        push!(closing_faces, closing_faces_local)

        return length(faces) # Index of the new node
    end

    determine_closure!(closing_faces_local, min_corner_local, max_corner_local, geometry,
                       face_ids, directed_edges)

    if length(closing_faces_local) >= length(face_ids)
        push!(faces, faces_to_tuple(face_ids, geometry))
        push!(min_corners, min_corner_local)
        push!(max_corners, max_corner_local)
        push!(childs_left, -1)
        push!(childs_right, -1)
        push!(closing_faces, closing_faces_local)

        return length(faces) # Index of the new node
    end

    # Bisect the box splitting its longest side
    box_edges = max_corner_local - min_corner_local

    split_direction = argmax(box_edges)

    uvec = (1:NDIMS) .== split_direction

    max_corner_left = max_corner_local - box_edges[split_direction] / 2 * uvec
    min_corner_right = min_corner_local + box_edges[split_direction] / 2 * uvec

    faces_left = is_in_box(geometry, face_ids, min_corner_local, max_corner_left)
    faces_right = is_in_box(geometry, face_ids, min_corner_right, max_corner_local)

    left_index = build_tree!(faces, closing_faces, min_corners, max_corners,
                             childs_left, childs_right, directed_edges,
                             geometry, faces_left, min_corner_local, max_corner_left)
    right_index = build_tree!(faces, closing_faces, min_corners, max_corners,
                              childs_left, childs_right, directed_edges,
                              geometry, faces_right, min_corner_right, max_corner_local)

    push!(faces, faces_to_tuple(face_ids, geometry))
    push!(closing_faces, closing_faces_local)
    push!(min_corners, min_corner_local)
    push!(max_corners, max_corner_local)
    push!(childs_left, left_index)
    push!(childs_right, right_index)

    return length(faces) # Index of the new node
end

function faces_to_tuple(edge_ids, geometry::Polygon)
    (; edge_vertices_ids) = geometry

    return map(i -> edge_vertices_ids[i], edge_ids)
end

function faces_to_tuple(face_ids, geometry::TriangleMesh)
    (; face_vertices_ids) = geometry

    return map(i -> face_vertices_ids[i], face_ids)
end

struct HierarchicalWinding{T}
    bounding_box_tree::T
end

function HierarchicalWinding(geometry::Geometry)
    return HierarchicalWinding(BoundingBoxTree(geometry))
end

@inline function (winding::HierarchicalWinding)(geometry, query_point)
    (; bounding_box_tree) = winding

    return hierarchical_winding(bounding_box_tree, bounding_box_tree.depth,
                                geometry, query_point)
end

function hierarchical_winding(tree::BoundingBoxTree, node_id, geometry, query_point)
    faces = tree.faces[node_id]
    closing_faces = tree.closing_faces[node_id]
    min_corner = tree.min_corners[node_id]
    max_corner = tree.max_corners[node_id]
    child_left = tree.childs_left[node_id]
    child_right = tree.childs_right[node_id]

    if child_left < 0
        # node is a leaf
        return naive_winding(geometry, faces, query_point)
    elseif !is_in_box(query_point, min_corner, max_corner)
        # `query_point` is outside bounding box
        return -naive_winding(geometry, closing_faces, query_point)
    end

    winding_number_left = hierarchical_winding(tree, child_left,
                                               geometry, query_point)
    winding_number_right = hierarchical_winding(tree, child_right,
                                                geometry, query_point)

    return winding_number_left + winding_number_right
end

# This only works when all `vertices` are unique
function determine_closure!(closing_faces, min_corner, max_corner, mesh::TriangleMesh,
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

            # Remove intersecting faces in the calculation of the exterior edges.
            # This way, all exterior edges are inside the bounding box, and so will be the closing surface.
            # The intersecting faces are later added with opposite orientation,
            # so that the closing is actually a closing of the exterior plus intersecting faces.
            push!(intersecting_faces, face)
        end
    end

    exterior_edges = findall(!iszero, directed_edges)

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

    # See comment above as to why intersecting faces are treated separately
    for face in intersecting_faces
        v1 = face_vertices_ids[face][1]
        v2 = face_vertices_ids[face][2]
        v3 = face_vertices_ids[face][3]

        # Flip order of vertices
        push!(closing_faces, (v3, v2, v1))
    end

    return closing_faces
end

# This only works when all `vertices` are unique
function determine_closure!(closing_edges, min_corner, max_corner, polygon::Polygon,
                            edges, vertex_count)
    (; edge_vertices_ids, edge_vertices) = polygon

    vertex_count .= 0

    intersecting_edges = Int[]

    # Find all exterior vertices
    for edge in edges
        if is_in_box(edge_vertices[edge][1], min_corner, max_corner) &&
           is_in_box(edge_vertices[edge][2], min_corner, max_corner)
            # Edge is completely inside the bounding box

            v1 = edge_vertices_ids[edge][1]
            v2 = edge_vertices_ids[edge][2]

            vertex_count[v1] += 1
            vertex_count[v2] -= 1
        else # Edge is intersecting the boundaries

            # Remove intersecting edges in the calculation of the exterior vertices.
            # This way, all exterior vertices are inside the bounding box,
            # and so will be the closing surface.
            # The intersecting edges are later added with opposite orientation,
            # so that the closing is actually a closing of the exterior plus intersecting edges.
            push!(intersecting_edges, edge)
        end
    end

    exterior_vertices = findall(!iszero, vertex_count)
    resize!(closing_edges, 0)

    if !isempty(exterior_vertices)
        closing_vertex = first(exterior_vertices)
    end

    for i in eachindex(exterior_vertices)[2:end]
        v1 = exterior_vertices[i]

        # Check orientation
        # `vertex_count[v1] > 0`: `v1` was start-vertex of the edge
        # `vertex_count[v1] < 0`: `v1` was end-vertex of the edge
        edge = vertex_count[v1] > 0 ? (closing_vertex, v1) : (v1, closing_vertex)

        push!(closing_edges, edge)
    end

    # See comment above as to why intersecting edges are treated separately
    for edge in intersecting_edges
        v1 = edge_vertices_ids[edge][1]
        v2 = edge_vertices_ids[edge][2]

        # Flip order of vertices
        push!(closing_edges, (v2, v1))
    end

    return closing_edges
end

function is_in_box(mesh, faces, min_corner, max_corner)
    return filter(face -> is_in_box(barycenter(mesh, face), min_corner, max_corner), faces)
end

@inline function barycenter(mesh::Polygon, edge)
    (; edge_vertices) = mesh

    v1 = edge_vertices[edge][1]
    v2 = edge_vertices[edge][2]

    return 0.5(v1 + v2)
end

@inline function barycenter(mesh::TriangleMesh, face)
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

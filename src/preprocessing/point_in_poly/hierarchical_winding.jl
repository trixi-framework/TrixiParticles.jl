# This bounding box is used for the hierarchical evaluation of the `WindingNumberJacobsen`.
# It is implementing a binary tree and thus stores the left and right child and also the
# faces and closing faces which are inside the bounding box.
struct BoundingBoxTree{MC, NDIMS}
    faces::Vector{NTuple{NDIMS, Int}}
    closing_faces::Vector{NTuple{NDIMS, Int}}
    min_corner::MC
    max_corner::MC
    is_leaf::Bool
    child_left::BoundingBoxTree
    child_right::BoundingBoxTree

    function BoundingBoxTree(geometry, face_ids, directed_edges, min_corner, max_corner)
        closing_faces = Vector{NTuple{ndims(geometry), Int}}()

        max_faces_in_box = ndims(geometry) == 3 ? 100 : 20
        if length(face_ids) < max_faces_in_box
            return new{
                typeof(min_corner),
                ndims(geometry),
            }(
                faces(face_ids, geometry), closing_faces,
                min_corner, max_corner, true
            )
        end

        determine_closure!(
            closing_faces, min_corner, max_corner, geometry, face_ids,
            directed_edges
        )

        if length(closing_faces) >= length(face_ids)
            return new{
                typeof(min_corner),
                ndims(geometry),
            }(
                faces(face_ids, geometry), closing_faces,
                min_corner, max_corner, true
            )
        end

        # Bisect the box splitting its longest side
        box_edges = max_corner - min_corner

        split_direction = argmax(box_edges)

        uvec = (1:ndims(geometry)) .== split_direction

        max_corner_left = max_corner - 0.5box_edges[split_direction] * uvec
        min_corner_right = min_corner + 0.5box_edges[split_direction] * uvec

        faces_left = is_in_box(geometry, face_ids, min_corner, max_corner_left)
        faces_right = is_in_box(geometry, face_ids, min_corner_right, max_corner)

        child_left = BoundingBoxTree(
            geometry, faces_left, directed_edges,
            min_corner, max_corner_left
        )
        child_right = BoundingBoxTree(
            geometry, faces_right, directed_edges,
            min_corner_right, max_corner
        )

        return new{
            typeof(min_corner),
            ndims(geometry),
        }(
            faces(face_ids, geometry), closing_faces,
            min_corner, max_corner, false, child_left, child_right
        )
    end
end

function faces(edge_ids, geometry::Polygon)
    (; edge_vertices_ids) = geometry

    return map(i -> edge_vertices_ids[i], edge_ids)
end

function faces(face_ids, geometry::TriangleMesh)
    (; face_vertices_ids) = geometry

    return map(i -> face_vertices_ids[i], face_ids)
end

struct HierarchicalWinding{BB}
    bounding_box::BB

    function HierarchicalWinding(geometry)
        # Note that overlapping bounding boxes are perfectly fine
        min_corner = geometry.min_corner .- sqrt(eps())
        max_corner = geometry.max_corner .+ sqrt(eps())

        if ndims(geometry) == 3
            directed_edges = zeros(Int, length(geometry.edge_normals))
        else
            directed_edges = zeros(Int, length(geometry.vertices))
        end

        bounding_box = BoundingBoxTree(
            geometry, eachface(geometry), directed_edges,
            min_corner, max_corner
        )

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
function determine_closure!(
        closing_faces, min_corner, max_corner, mesh::TriangleMesh{3},
        faces, directed_edges
    )
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
function determine_closure!(
        closing_edges, min_corner, max_corner, polygon::Polygon{2},
        edges, vertex_count
    )
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

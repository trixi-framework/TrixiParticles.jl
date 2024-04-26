struct TriangleMesh{NDIMS, ELTYPE} <: Shapes{NDIMS}
    face_vertices  :: Vector{Vector{SVector{NDIMS, ELTYPE}}}
    normals_vertex :: Vector{Vector{SVector{NDIMS, ELTYPE}}}
    normals_edge   :: Vector{Vector{SVector{NDIMS, ELTYPE}}}
    normals_face   :: Vector{SVector{NDIMS, ELTYPE}}
    min_box        :: SVector{NDIMS, ELTYPE}
    max_box        :: SVector{NDIMS, ELTYPE}

    function TriangleMesh(mesh)
        NDIMS = length(first(mesh))
        n_faces = length(mesh)

        min_box = SVector([minimum(v[i] for v in mesh.position) for i in 1:NDIMS]...)
        max_box = SVector([maximum(v[i] for v in mesh.position) for i in 1:NDIMS]...)

        ELTYPE = eltype(min_box)

        triangle_connectivity = decompose(TriangleFace{Int}, mesh)

        point_per_edge_indices, edges_in_triangle = populate_edges(triangle_connectivity)

        face_vertices = [[fill(zero(ELTYPE), SVector{NDIMS}) for _ in 1:3]
                         for _ in 1:n_faces]
        normals_face = [fill(zero(ELTYPE), SVector{NDIMS}) for _ in 1:n_faces]

        normals_edge = [fill(zero(ELTYPE), SVector{NDIMS}) for _ in 1:(3n_faces)]
        normals_vertex = [fill(zero(ELTYPE), SVector{NDIMS}) for _ in 1:(3n_faces)]

        for i in 1:n_faces
            v1, v2, v3 = mesh[i]

            # Calculate normals
            n = SVector(normalize(cross(v2 - v1, v3 - v1))...)

            face_vertices[i] = [SVector(v1...), SVector(v2...), SVector(v3...)]
            normals_face[i] = n

            # Compute edge and point pseudo-normals
            # See Baerentzen and Aanaes (2002)
            for edge_index in edges_in_triangle[i]
                normals_edge[edge_index] += n
            end

            angles = incident_angles(face_vertices[i])

            j = 1
            for vertex_index in triangle_connectivity[i]
                normals_vertex[vertex_index] += angles[j] * n
                j += 1
            end
        end

        # Reshape
        normals_edge_ = [normalize.(normals_edge[edges_in_triangle[i]]) for i in 1:n_faces]
        normals_vertex_ = [normalize.(normals_vertex[edges_in_triangle[i]])
                           for i in 1:n_faces]

        return new{NDIMS, ELTYPE}(face_vertices, normals_vertex_, normals_edge_,
                                  normals_face, min_box, max_box)
    end
end

@inline nfaces(mesh::TriangleMesh) = length(mesh.normals_face)

# Copied from https://gitlab.emse.fr/pierrat/SignedDistanceField.jl/-/blob/master/src/meshutils.jl?ref_type=heads#L71
# Generates edge connectivity from triangle connectivity
function populate_edges(trisconn)
    edges = Vector{Tuple{Int, Int}}()
    edge_idx = 0
    edges_in_tri = [[0, 0, 0] for _ in 1:length(trisconn)]

    for (idx, t) in enumerate(trisconn)
        for i in 1:3
            pt1 = t[i]
            pt2 = t[mod1(i + 1, 3)]

            edge = (min(pt1, pt2), max(pt1, pt2))

            edge_idx_a = findfirst(isequal(edge), edges)
            if isnothing(edge_idx_a)
                push!(edges, edge)
                edge_idx = length(edges)
            else
                edge_idx = edge_idx_a
            end

            edges_in_tri[idx][i] = edge_idx
        end
    end

    return edges, edges_in_tri
end

# Almost identical to https://gitlab.emse.fr/pierrat/SignedDistanceField.jl/-/blob/master/src/SignedDistanceField.jl?ref_type=heads#L15
function incident_angles(triangle_points)
    a = triangle_points[1]
    b = triangle_points[2]
    c = triangle_points[3]
    sab = dot(b - a, b - a)
    sbc = dot(b - c, b - c)
    sac = dot(c - a, c - a)

    alpha = acos(clamp((sab + sac - sbc) / (2 * sqrt(sab * sac)), -1, 1))
    beta = acos(clamp((sbc + sab - sac) / (2 * sqrt(sbc * sab)), -1, 1))
    gamma = acos(clamp((sac + sbc - sab) / (2 * sqrt(sac * sbc)), -1, 1))

    return alpha, beta, gamma
end

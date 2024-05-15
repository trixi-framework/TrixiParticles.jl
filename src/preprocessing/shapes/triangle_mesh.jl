struct TriangleMesh{NDIMS, ELTYPE} <: Shapes{NDIMS}
    vertices          :: Vector{SVector{NDIMS, ELTYPE}}
    face_vertices     :: Vector{Tuple{SVector{NDIMS, ELTYPE}, SVector{NDIMS, ELTYPE}, SVector{NDIMS, ELTYPE}}}
    face_vertices_ids :: Vector{NTuple{3, Int}}
    face_edges_ids    :: Vector{NTuple{3, Int}}
    edge_vertices_ids :: Vector{NTuple{2, Int}}
    normals_vertex    :: Vector{SVector{NDIMS, ELTYPE}}
    normals_edge      :: Vector{SVector{NDIMS, ELTYPE}}
    normals_face      :: Vector{SVector{NDIMS, ELTYPE}}
    min_box           :: SVector{NDIMS, ELTYPE}
    max_box           :: SVector{NDIMS, ELTYPE}

    function TriangleMesh(face_vertices, normals_face, vertices)
        NDIMS = 3
        n_faces = length(normals_face)

        min_box = SVector([minimum(v[i] for v in vertices) for i in 1:NDIMS]...)
        max_box = SVector([maximum(v[i] for v in vertices) for i in 1:NDIMS]...)

        ELTYPE = eltype(first(normals_face))

        face_vertices_ids = [(0, 0, 0) for _ in 1:n_faces]
        face_edges_ids = [(0, 0, 0) for _ in 1:n_faces]

        _edges = Dict{NTuple{2, Int}, Int}()

        normals_vertex = [fill(zero(ELTYPE), SVector{NDIMS}) for _ in 1:length(vertices)]

        @threaded for i in 1:n_faces
            v1 = face_vertices[i][1]
            v2 = face_vertices[i][2]
            v3 = face_vertices[i][3]

            vertex_id1 = findfirst(x -> v1 == x, vertices)
            vertex_id2 = findfirst(x -> v2 == x, vertices)
            vertex_id3 = findfirst(x -> v3 == x, vertices)

            face_vertices_ids[i] = (vertex_id1, vertex_id2, vertex_id3)
        end

        _edges = Dict{NTuple{2, Int}, Int}()
        normals_edge = [fill(zero(ELTYPE), SVector{NDIMS}) for _ in 1:(3n_faces)]
        edge_vertices_ids = fill((0, 0), 3n_faces)

        # Not thread supported (yet)
        edge_id = 0
        for i in 1:n_faces
            v1 = face_vertices_ids[i][1]
            v2 = face_vertices_ids[i][2]
            v3 = face_vertices_ids[i][3]

            if haskey(_edges, (v1, v2))
                edge_id_1 = _edges[(v1, v2)]
            elseif haskey(_edges, (v2, v1))
                edge_id_1 = _edges[(v2, v1)]
            else
                edge_id += 1
                _edges[(v1, v2)] = edge_id
                edge_id_1 = edge_id
            end
            edge_vertices_ids[edge_id_1] = (v1, v2)

            if haskey(_edges, (v2, v3))
                edge_id_2 = _edges[(v2, v3)]
            elseif haskey(_edges, (v3, v2))
                edge_id_2 = _edges[(v3, v2)]
            else
                edge_id += 1
                _edges[(v2, v3)] = edge_id
                edge_id_2 = edge_id
            end
            edge_vertices_ids[edge_id_2] = (v2, v3)

            if haskey(_edges, (v3, v1))
                edge_id_3 = _edges[(v3, v1)]
            elseif haskey(_edges, (v1, v3))
                edge_id_3 = _edges[(v1, v3)]
            else
                edge_id += 1
                _edges[(v3, v1)] = edge_id
                edge_id_3 = edge_id
            end
            edge_vertices_ids[edge_id_3] = (v3, v1)

            face_edges_ids[i] = (edge_id_1, edge_id_2, edge_id_3)

            normals_edge[edge_id_1] += normals_face[i]
            normals_edge[edge_id_2] += normals_face[i]
            normals_edge[edge_id_3] += normals_face[i]

            angles = incident_angles(face_vertices[i])

            normals_vertex[v1] += angles[1] * normals_face[i]
            normals_vertex[v2] += angles[2] * normals_face[i]
            normals_vertex[v3] += angles[3] * normals_face[i]
        end

        resize!(normals_edge, length(_edges))
        resize!(edge_vertices_ids, length(_edges))

        return new{NDIMS, ELTYPE}(vertices, face_vertices, face_vertices_ids,
                                  face_edges_ids, edge_vertices_ids,
                                  normalize.(normals_vertex), normalize.(normals_edge),
                                  normals_face, min_box, max_box)
    end
end

@inline nfaces(mesh::TriangleMesh) = length(mesh.normals_face)

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

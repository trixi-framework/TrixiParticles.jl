struct TriangleMesh{NDIMS, ELTYPE} <: Shapes{NDIMS}
    face_vertices     :: Vector{Tuple{SVector{NDIMS, ELTYPE}, SVector{NDIMS, ELTYPE}, SVector{NDIMS, ELTYPE}}}
    face_vertices_ids :: Vector{NTuple{3, Int}}
    face_edges_ids    :: Vector{NTuple{3, Int}}
    normals_vertex    :: Vector{SVector{NDIMS, ELTYPE}}
    normals_edge      :: Vector{SVector{NDIMS, ELTYPE}}
    normals_face      :: Vector{SVector{NDIMS, ELTYPE}}
    min_box           :: SVector{NDIMS, ELTYPE}
    max_box           :: SVector{NDIMS, ELTYPE}

    function TriangleMesh(mesh)
        NDIMS = length(first(mesh))
        n_faces = length(mesh)

        min_box = SVector([minimum(v[i] for v in mesh.position) for i in 1:NDIMS]...)
        max_box = SVector([maximum(v[i] for v in mesh.position) for i in 1:NDIMS]...)

        ELTYPE = eltype(min_box)

        vertices = union(mesh.position)

        face_vertices_ids = [(0, 0, 0) for _ in 1:n_faces]
        face_edges_ids = [(0, 0, 0) for _ in 1:n_faces]

        _edges = Dict{NTuple{2, Int}, Int}()

        face_vertices = [ntuple(_ -> SVector(0.0, 0.0, 0.0), 3) for _ in 1:n_faces]

        normals_face = [fill(zero(ELTYPE), SVector{NDIMS}) for _ in 1:n_faces]
        normals_vertex = [fill(zero(ELTYPE), SVector{NDIMS}) for _ in 1:length(vertices)]

        @threaded for i in 1:n_faces
            v1, v2, v3 = mesh[i]
            vertex_id1 = findfirst(x -> v1 == x, vertices)
            vertex_id2 = findfirst(x -> v2 == x, vertices)
            vertex_id3 = findfirst(x -> v3 == x, vertices)

            face_vertices_ids[i] = (vertex_id1, vertex_id2, vertex_id3)
            face_vertices[i] = (vertices[vertex_id1], vertices[vertex_id2],
                                vertices[vertex_id3])

            # Calculate normals
            n = normalize(cross(vertices[vertex_id2] - vertices[vertex_id1],
                                vertices[vertex_id3] - vertices[vertex_id1]))

            normals_face[i] = n

            normals_vertex[vertex_id1] += n
            normals_vertex[vertex_id2] += n
            normals_vertex[vertex_id3] += n
        end

        _edges = Dict{NTuple{2, Int}, Int}()
        normals_edge = [fill(zero(ELTYPE), SVector{NDIMS}) for _ in 1:(3n_faces)]

        # Not thread supported (yet)
        edge_id = 0
        for i in 1:n_faces
            vertex_id1 = face_vertices_ids[i][1]
            vertex_id2 = face_vertices_ids[i][2]
            vertex_id3 = face_vertices_ids[i][3]

            edge_1 = (vertex_id1, vertex_id2)
            edge_1_ = (vertex_id2, vertex_id1)
            edge_2 = (vertex_id2, vertex_id3)
            edge_2_ = (vertex_id3, vertex_id2)
            edge_3 = (vertex_id3, vertex_id1)
            edge_3_ = (vertex_id1, vertex_id3)

            if haskey(_edges, edge_1)
                edge_id_1 = _edges[edge_1]
            elseif haskey(_edges, edge_1_)
                edge_id_1 = _edges[edge_1_]
            else
                edge_id += 1
                _edges[edge_1] = edge_id
                edge_id_1 = edge_id
            end

            if haskey(_edges, edge_2)
                edge_id_2 = _edges[edge_2]
            elseif haskey(_edges, edge_2_)
                edge_id_2 = _edges[edge_2_]
            else
                edge_id += 1
                _edges[edge_2] = edge_id
                edge_id_2 = edge_id
            end

            if haskey(_edges, edge_3)
                edge_id_3 = _edges[edge_3]
            elseif haskey(_edges, edge_3_)
                edge_id_3 = _edges[edge_3_]
            else
                edge_id += 1
                _edges[edge_3] = edge_id
                edge_id_3 = edge_id
            end

            face_edges_ids[i] = (edge_id_1, edge_id_2, edge_id_3)

            normals_edge[edge_id_1] += normals_face[i]
            normals_edge[edge_id_2] += normals_face[i]
            normals_edge[edge_id_3] += normals_face[i]
        end

        resize!(normals_edge, length(_edges))

        return new{NDIMS, ELTYPE}(face_vertices, face_vertices_ids, face_edges_ids,
                                  normalize.(normals_vertex), normalize.(normals_edge),
                                  normals_face, min_box, max_box)
    end
end

@inline nfaces(mesh::TriangleMesh) = length(mesh.normals_face)

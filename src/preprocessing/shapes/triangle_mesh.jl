# This is the data format returned by `load(file)` when used with `.stl` files
struct TriangleMesh{NDIMS, ELTYPE}
    vertices          :: Vector{SVector{NDIMS, ELTYPE}}
    face_vertices     :: Vector{NTuple{3, SVector{NDIMS, ELTYPE}}}
    face_vertices_ids :: Vector{NTuple{3, Int}}
    face_edges_ids    :: Vector{NTuple{3, Int}}
    edge_vertices_ids :: Vector{NTuple{2, Int}}
    vertex_normals    :: Vector{SVector{NDIMS, ELTYPE}}
    edge_normals      :: Vector{SVector{NDIMS, ELTYPE}}
    face_normals      :: Vector{SVector{NDIMS, ELTYPE}}
    min_corner        :: SVector{NDIMS, ELTYPE}
    max_corner        :: SVector{NDIMS, ELTYPE}

    function TriangleMesh(face_vertices, face_normals, vertices)
        NDIMS = 3
        ELTYPE = eltype(first(face_normals))
        n_faces = length(face_normals)

        face_vertices_ids = fill((0, 0, 0), n_faces)

        @threaded face_vertices for i in 1:n_faces
            v1 = face_vertices[i][1]
            v2 = face_vertices[i][2]
            v3 = face_vertices[i][3]

            # TODO: This part is about 90% of the runtime.
            # Vertex IDs are only needed for the hierarchical winding.
            vertex_id1 = findfirst(x -> v1 == x, vertices)
            vertex_id2 = findfirst(x -> v2 == x, vertices)
            vertex_id3 = findfirst(x -> v3 == x, vertices)

            face_vertices_ids[i] = (vertex_id1, vertex_id2, vertex_id3)
        end

        _edges = Dict{NTuple{2, Int}, Int}()
        face_edges_ids = fill((0, 0, 0), n_faces)
        edge_normals = fill(fill(zero(ELTYPE), SVector{NDIMS}), 3n_faces)
        vertex_normals = fill(fill(zero(ELTYPE), SVector{NDIMS}), length(vertices))
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

            edge_normals[edge_id_1] += face_normals[i]
            edge_normals[edge_id_2] += face_normals[i]
            edge_normals[edge_id_3] += face_normals[i]

            angles = incident_angles(face_vertices[i])

            vertex_normals[v1] += angles[1] * face_normals[i]
            vertex_normals[v2] += angles[2] * face_normals[i]
            vertex_normals[v3] += angles[3] * face_normals[i]
        end

        resize!(edge_normals, length(_edges))
        resize!(edge_vertices_ids, length(_edges))

        min_corner = SVector([minimum(v[i] for v in vertices) for i in 1:NDIMS]...)
        max_corner = SVector([maximum(v[i] for v in vertices) for i in 1:NDIMS]...)

        return new{NDIMS, ELTYPE}(vertices, face_vertices, face_vertices_ids,
                                  face_edges_ids, edge_vertices_ids,
                                  normalize.(vertex_normals), normalize.(edge_normals),
                                  face_normals, min_corner, max_corner)
    end
end

@inline Base.ndims(::TriangleMesh{NDIMS}) where {NDIMS} = NDIMS

@inline nfaces(mesh::TriangleMesh) = length(mesh.face_normals)

@inline function face_vertices(triangle, shape::TriangleMesh)
    v1 = shape.face_vertices[triangle][1]
    v2 = shape.face_vertices[triangle][2]
    v3 = shape.face_vertices[triangle][3]

    return v1, v2, v3
end

function incident_angles(triangle_points)
    a = triangle_points[1]
    b = triangle_points[2]
    c = triangle_points[3]
    sab = dot(b - a, b - a)
    sbc = dot(b - c, b - c)
    sac = dot(c - a, c - a)

    alpha_ = (sab + sac - sbc) / (2 * sqrt(sab * sac))
    beta_ = (sbc + sab - sac) / (2 * sqrt(sbc * sab))
    gamma_ = (sac + sbc - sab) / (2 * sqrt(sac * sbc))

    alpha_ = isfinite(alpha_) ? clamp(alpha_, -1, 1) : zero(eltype(sab))
    beta_ = isfinite(beta_) ? clamp(beta_, -1, 1) : zero(eltype(sab))
    gamma_ = isfinite(gamma_) ? clamp(gamma_, -1, 1) : zero(eltype(sab))

    alpha = acos(alpha_)
    beta = acos(beta_)
    gamma = acos(gamma_)

    return alpha, beta, gamma
end

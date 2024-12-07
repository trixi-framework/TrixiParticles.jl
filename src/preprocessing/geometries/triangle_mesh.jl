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

        return TriangleMesh{NDIMS}(face_vertices, face_normals, vertices)
    end

    # Function barrier to make `NDIMS` static and therefore `SVector`s type-stable
    function TriangleMesh{NDIMS}(face_vertices, face_normals, vertices_) where {NDIMS}
        # Sort vertices by the first entry of the vector and return only unique vertices
        vertices = unique_sorted(vertices_)

        ELTYPE = eltype(first(face_normals))
        n_faces = length(face_normals)

        face_vertices_ids = fill((0, 0, 0), n_faces)

        @threaded face_vertices for i in 1:n_faces
            v1 = face_vertices[i][1]
            v2 = face_vertices[i][2]
            v3 = face_vertices[i][3]

            # Since it's only sorted by the first entry, `v1` might be one of the following vertices
            vertex_id1 = searchsortedfirst(vertices, v1 .- 1e-14)
            for vertex_id in eachindex(vertices)[vertex_id1:end]
                if isapprox(vertices[vertex_id], v1)
                    vertex_id1 = vertex_id
                    break
                end
            end

            # Since it's only sorted by the first entry, `v2` might be one of the following vertices
            vertex_id2 = searchsortedfirst(vertices, v2 .- 1e-14)
            for vertex_id in eachindex(vertices)[vertex_id2:end]
                if isapprox(vertices[vertex_id], v2)
                    vertex_id2 = vertex_id
                    break
                end
            end

            # Since it's only sorted by the first entry, `v3` might be one of the following vertices
            vertex_id3 = searchsortedfirst(vertices, v3 .- 1e-14)
            for vertex_id in eachindex(vertices)[vertex_id3:end]
                if isapprox(vertices[vertex_id], v3)
                    vertex_id3 = vertex_id
                    break
                end
            end

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

            # Make sure that edges are unique
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

            # Edge normal is the sum of the normals of the two adjacent faces
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

@inline Base.eltype(::TriangleMesh{NDIMS, ELTYPE}) where {NDIMS, ELTYPE} = ELTYPE

@inline face_normal(triangle, geometry::TriangleMesh) = geometry.face_normals[triangle]

@inline function Base.deleteat!(mesh::TriangleMesh, indices)
    (; face_vertices, face_vertices_ids, face_edges_ids, face_normals) = mesh

    deleteat!(face_vertices, indices)
    deleteat!(face_vertices_ids, indices)
    deleteat!(face_edges_ids, indices)
    deleteat!(face_normals, indices)

    return mesh
end

@inline nfaces(mesh::TriangleMesh) = length(mesh.face_normals)

@inline function face_vertices(triangle, geometry::TriangleMesh)
    v1 = geometry.face_vertices[triangle][1]
    v2 = geometry.face_vertices[triangle][2]
    v3 = geometry.face_vertices[triangle][3]

    return v1, v2, v3
end

function incident_angles(triangle_points)
    a = triangle_points[1]
    b = triangle_points[2]
    c = triangle_points[3]

    # Squares of the lengths of the sides
    ab2 = dot(b - a, b - a)
    bc2 = dot(b - c, b - c)
    ac2 = dot(c - a, c - a)

    # Applying the law of cosines
    # https://en.wikipedia.org/wiki/Law_of_cosines
    cos_alpha = (ab2 + ac2 - bc2) / (2 * sqrt(ab2 * ac2))
    cos_beta = (bc2 + ab2 - ac2) / (2 * sqrt(bc2 * ab2))
    cos_gamma = (ac2 + bc2 - ab2) / (2 * sqrt(ac2 * bc2))

    # If one side has length zero, this assures that the two adjacent angles
    # become pi/2, while the third angle becomes zero.
    cos_alpha = isfinite(cos_alpha) ? clamp(cos_alpha, -1, 1) : zero(eltype(ab2))
    cos_beta = isfinite(cos_beta) ? clamp(cos_beta, -1, 1) : zero(eltype(ab2))
    cos_gamma = isfinite(cos_gamma) ? clamp(cos_gamma, -1, 1) : zero(eltype(ab2))

    alpha = acos(cos_alpha)
    beta = acos(cos_beta)
    gamma = acos(cos_gamma)

    return alpha, beta, gamma
end

function unique_sorted(vertices)
    # Sort by the first entry of the vectors
    compare_first_element = (x, y) -> x[1] < y[1]
    vertices_sorted = sort!(vertices, lt=compare_first_element)
    # We cannot use a `BitVector` here, as writing to a `BitVector` is not thread-safe
    keep = fill(true, length(vertices_sorted))

    PointNeighbors.@threaded vertices_sorted for i in eachindex(vertices_sorted)
        # We only sorted by the first entry, so we have to check all previous vertices
        # until the first entry is too far away.
        j = i - 1
        while j >= 1 && isapprox(vertices_sorted[j][1], vertices_sorted[i][1], atol=1e-14)
            if isapprox(vertices_sorted[i], vertices_sorted[j], atol=1e-14)
                keep[i] = false
            end
            j -= 1
        end
    end

    return vertices_sorted[keep]
end

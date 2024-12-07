# This is the data format returned by `load(file)` when used with `.asc` files
struct Polygon{NDIMS, ELTYPE}
    vertices          :: Vector{SVector{NDIMS, ELTYPE}}
    edge_vertices     :: Vector{NTuple{2, SVector{NDIMS, ELTYPE}}}
    vertex_normals    :: Vector{NTuple{2, SVector{NDIMS, ELTYPE}}}
    edge_normals      :: Vector{SVector{NDIMS, ELTYPE}}
    edge_vertices_ids :: Vector{NTuple{2, Int}}
    min_corner        :: SVector{NDIMS, ELTYPE}
    max_corner        :: SVector{NDIMS, ELTYPE}

    function Polygon(vertices)
        NDIMS = size(vertices, 1)

        return Polygon{NDIMS}(vertices)
    end

    # Function barrier to make `NDIMS` static and therefore `SVector`s type-stable
    function Polygon{NDIMS}(vertices_) where {NDIMS}
        n_vertices = size(vertices_, 2)
        ELTYPE = eltype(vertices_)

        min_corner = SVector{NDIMS}(minimum(vertices_, dims=2))
        max_corner = SVector{NDIMS}(maximum(vertices_, dims=2))

        vertices = reinterpret(reshape, SVector{NDIMS, ELTYPE}, vertices_)

        # Sum over all the edges and determine if the vertices are in clockwise order
        # to make sure that all normals pointing outwards.
        # To do so, we compute the signed area of the polygon by the shoelace formula, which
        # is positive for counter-clockwise ordering and negative for clockwise ordering of the vertices.
        # https://en.wikipedia.org/wiki/Polygon
        polygon_area = 0.0
        for i in 1:(n_vertices - 1)
            v1 = vertices[i]
            v2 = vertices[i + 1]
            polygon_area += (v1[1] * v2[2] - v2[1] * v1[2])
        end

        if polygon_area == 0.0
            throw(ArgumentError("polygon is not correctly defined"))
        elseif polygon_area > 0.0
            # Curve is counter-clockwise
            reverse!(vertices)
        end

        edge_vertices = Vector{NTuple{2, SVector{NDIMS, ELTYPE}}}()
        edge_vertices_ids = Vector{NTuple{2, Int}}()
        edge_normals = Vector{SVector{NDIMS, ELTYPE}}()

        for i in 1:(n_vertices - 1)
            v1 = vertices[i]
            v2 = vertices[i + 1]
            if isapprox(v1, v2)
                continue
            end

            edge = v2 - v1

            edge_normal = SVector{NDIMS}(normalize([-edge[2], edge[1]]))

            push!(edge_vertices, (v1, v2))
            push!(edge_vertices_ids, (i, i + 1))
            push!(edge_normals, edge_normal)
        end

        vertex_normals = Vector{NTuple{2, SVector{NDIMS, ELTYPE}}}()

        # Calculate vertex pseudo-normals.
        # An edge is defined by two vertices, for which we calculate the pseudo-normals
        # by averaging the normals of the adjacent edges.
        # We calculate the distances for a single edge in `calculate_signed_distances!`
        # and don't look for adjacent edges. Thus, we store the pseudo-normals of the vertices
        # per edge.
        for i in 1:length(edge_vertices)
            if i == 1
                edge_normal_1 = edge_normals[end]
            else
                edge_normal_1 = edge_normals[i - 1]
            end

            edge_normal_2 = edge_normals[i]

            if i == length(edge_vertices)
                edge_normal_3 = edge_normals[1]
            else
                edge_normal_3 = edge_normals[i + 1]
            end

            vortex_normal_1 = normalize(edge_normal_1 + edge_normal_2)
            vortex_normal_2 = normalize(edge_normal_2 + edge_normal_3)

            push!(vertex_normals, (vortex_normal_1, vortex_normal_2))
        end

        return new{NDIMS, ELTYPE}(vertices, edge_vertices, vertex_normals, edge_normals,
                                  edge_vertices_ids, min_corner, max_corner)
    end
end

@inline Base.ndims(::Polygon{NDIMS}) where {NDIMS} = NDIMS

@inline Base.eltype(::Polygon{NDIMS, ELTYPE}) where {NDIMS, ELTYPE} = ELTYPE

@inline function Base.deleteat!(polygon::Polygon, indices)
    (; edge_vertices, edge_normals, edge_vertices_ids) = polygon

    deleteat!(edge_vertices, indices)
    deleteat!(edge_vertices_ids, indices)
    deleteat!(edge_normals, indices)

    return polygon
end

@inline nfaces(mesh::Polygon) = length(mesh.edge_normals)

@inline function face_vertices(edge, geometry::Polygon)
    v1 = geometry.edge_vertices[edge][1]
    v2 = geometry.edge_vertices[edge][2]

    return v1, v2
end

@inline face_normal(edge, geometry::Polygon) = geometry.edge_normals[edge]

struct Polygon{NDIMS, ELTYPE} <: Shapes{NDIMS}
    edge_vertices  :: Vector{Vector{SVector{NDIMS, ELTYPE}}}
    normals_vertex :: Vector{Vector{SVector{NDIMS, ELTYPE}}}
    normals_edge   :: Vector{SVector{NDIMS, ELTYPE}}
    min_corner     :: SVector{NDIMS, ELTYPE}
    max_corner     :: SVector{NDIMS, ELTYPE}

    function Polygon(vertices_)
        NDIMS = size(vertices_, 1)

        vertices_x = copy(vertices_[1, :])
        vertices_y = copy(vertices_[2, :])

        if !isapprox(vertices_[:, end], vertices_[:, 1])
            push!(vertices_x, vertices_x[1])
            push!(vertices_y, vertices_y[1])

            vertices = vcat(vertices_x', vertices_y')
        else
            vertices = vertices_
        end

        min_corner = SVector([minimum(vertices[i, :]) for i in 1:NDIMS]...)
        max_corner = SVector([maximum(vertices[i, :]) for i in 1:NDIMS]...)

        n_vertices = size(vertices, 2)
        ELTYPE = eltype(vertices)

        # Sum over all the edges and determine if the vertices are in counter-clockwise order
        # to make sure that all normals pointing outwards
        counter = 0.0
        for i in 1:(n_vertices - 1)
            v1 = SVector(vertices[:, i]...)
            v2 = SVector(vertices[:, i + 1]...)
            counter += (v2[1] - v1[1]) * (v2[2] + v1[2])
        end

        if counter < 0.0
            # Curve is clockwise
            reverse!(vertices, dims=2)
            reverse!(vertices_x)
            reverse!(vertices_y)
        end

        edge_vertices = Vector{Vector{SVector{NDIMS, ELTYPE}}}()
        normals_edge = Vector{SVector{NDIMS, ELTYPE}}()

        for i in 1:(n_vertices - 1)
            v1 = SVector(vertices[:, i]...)
            v2 = SVector(vertices[:, i + 1]...)
            if isapprox(v1, v2)
                continue
            end

            edge = v2 - v1

            edge_normal = SVector(normalize([-edge[2], edge[1]])...)

            push!(edge_vertices, [v1, v2])
            push!(normals_edge, edge_normal)
        end

        normals_vertex = Vector{Vector{SVector{NDIMS, ELTYPE}}}()

        # Calculate vertex pseudo-normals

        for i in 1:length(edge_vertices)
            if i == 1
                edge_normal_1 = normals_edge[end]
            else
                edge_normal_1 = normals_edge[i - 1]
            end

            edge_normal_2 = normals_edge[i]

            if i == length(edge_vertices)
                edge_normal_3 = normals_edge[1]
            else
                edge_normal_3 = normals_edge[i + 1]
            end

            vortex_normal_1 = edge_normal_1 + edge_normal_2
            vortex_normal_2 = edge_normal_2 + edge_normal_3

            push!(normals_vertex, [vortex_normal_1, vortex_normal_2])
        end

        return new{NDIMS, ELTYPE}(edge_vertices, normals_vertex, normals_edge,
                                  min_corner, max_corner)
    end
end

@inline nfaces(mesh::Polygon) = length(mesh.normals_edge)

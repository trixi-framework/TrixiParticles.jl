struct Polygon{NDIMS, ELTYPE} <: Shapes{NDIMS}
    vertices   :: Array{ELTYPE, 2}
    edges      :: Array{ELTYPE, 3} # [dim, dim, face]
    n_vertices :: Int

    function Polygon(vertices_)
        NDIMS = size(vertices_, 1)

        if !(vertices_[:, end] ≈ vertices_[:, 1])
            vertices = zeros(NDIMS, size(vertices_, 2) + 1)
            vertices[:, 1:(end - 1)] = copy(vertices_)
            for dim in 1:NDIMS
                vertices[dim, end] = vertices_[dim, 1]
            end
        else
            vertices = copy(vertices_)
        end

        n_vertices = size(vertices, 2)
        ELTYPE = eltype(vertices)

        edges = zeros(ELTYPE, NDIMS, NDIMS, n_vertices - 1)

        for i in 1:(n_vertices - 1)
            v1 = position(vertices, Val(NDIMS), i)
            v2 = position(vertices, Val(NDIMS), i + 1)
            edges[:, 1, i] = v1
            edges[:, 2, i] = v2
        end

        return new{NDIMS, ELTYPE}(vertices, edges, n_vertices)
    end
end

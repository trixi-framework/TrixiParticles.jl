struct Polygon{NDIMS, ELTYPE} <: Shapes{NDIMS}
    vertices   :: Array{ELTYPE, 2}
    edges      :: Array{ELTYPE, 3} # [dim, dim, face]
    n_vertices :: Int

    function Polygon(vertices)
        if !(vertices[:, end] ≈ vertices[:, 1])
            error("The first and last vertex of the polygon must be the same.")
        end

        n_vertices = size(vertices, 2)
        ELTYPE = eltype(vertices)
        NDIMS = size(vertices, 1)

        edges = zeros(ELTYPE, NDIMS, NDIMS, n_vertices - 1)

        for i in 1:(n_vertices - 1)
            v1 = position(vertices, Val(2), i)
            v2 = position(vertices, Val(2), i + 1)
            edges[:, 1, i] = v1
            edges[:, 2, i] = v2
        end

        return new{NDIMS, ELTYPE}(vertices, edges, n_vertices)
    end
end

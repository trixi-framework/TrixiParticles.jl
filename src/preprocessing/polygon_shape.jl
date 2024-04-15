struct Polygon{NDIMS, ELTYPE, F} <: Shapes{NDIMS}
    vertices   :: Array{ELTYPE, 2}
    faces      :: F
    n_vertices :: Int

    function Polygon(vertices_)
        NDIMS = size(vertices_, 1)

        vertices_x = copy(vertices_[1, :])
        vertices_y = copy(vertices_[2, :])

        if !(vertices_[:, end] ≈ vertices_[:, 1])
            push!(vertices_x, vertices_x[1])
            push!(vertices_y, vertices_y[1])

            vertices = vcat(vertices_x', vertices_y')
        else
            vertices = vertices_
        end

        n_vertices = size(vertices, 2)
        ELTYPE = eltype(vertices)

        faces = Vector{NamedTuple}()

        delete_indices = Int[]

        for i in 1:(n_vertices - 1)
            v1 = SVector(vertices[:, i]...)
            v2 = SVector(vertices[:, i + 1]...)
            if isapprox(v1, v2)
                push!(delete_indices, i)
                continue
            end

            edge = v2 - v1

            # TODO: Recalculate all normals outside.
            edge_normal = SVector(-normalize([-edge[2], edge[1]])...)

            push!(faces, (v1=v1, v2=v2, normals=edge_normal))
        end

        deleteat!(vertices_x, delete_indices)
        deleteat!(vertices_y, delete_indices)

        vertices = vcat(vertices_x', vertices_y')

        return new{NDIMS, ELTYPE, typeof(faces)}(vertices, faces, length(faces))
    end
end

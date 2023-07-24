struct TriangleMesh{NDIMS, ELTYPE} <: Shapes{NDIMS}
    vertices   :: Array{ELTYPE, 2}
    normals    :: Array{ELTYPE, 2}
    faces      :: Array{ELTYPE, 3} # [dim, dim, face]
    n_vertices :: Int
    n_faces    :: Int

    function TriangleMesh(mesh)
        ELTYPE = eltype(first(mesh.position))
        NDIMS = length(first(mesh))
        NELEMENTS = length(mesh)

        normals = zeros(NDIMS, NELEMENTS)

        faces = zeros(ELTYPE, NDIMS, NDIMS, NELEMENTS)
        face_id = 0
        for face in mesh
            face_id += 1

            v1, v2, v3 = face
            a = v2 - v1
            b = v3 - v1
            normals[:, face_id] .= normalize(cross(a, b))
            faces[:, 1, face_id] = v1
            faces[:, 2, face_id] = v2
            faces[:, 3, face_id] = v3
        end

        vertices = stack(union(mesh.position))
        n_vertices = size(vertices, 2)

        return new{NDIMS, ELTYPE}(vertices, normals, faces, n_vertices, NELEMENTS)
    end
end

struct TriangleMesh{NDIMS, F, V} <: Shapes{NDIMS}
    faces           :: F
    unique_vertices :: V
    function TriangleMesh(faces)
        NDIMS = length(first(faces))

        # TODO: Recalculate all normals outside.
        # Or use blender function to do this and only check if all normals pointing outside.

        unique_vertices = stack(union(faces.position))
        return new{NDIMS, typeof(faces), typeof(unique_vertices)}(faces, unique_vertices)
    end
end

struct TriangleMesh{NDIMS, F, V} <: Shapes{NDIMS}
    faces      :: F
    vertices   :: V
    function TriangleMesh(faces)
        NDIMS = length(first(faces))

        # TODO: Recalculate all normals inside.
        # Or use blender function to do this and only check if all normals poinitng inside.

        vertices = stack(union(faces.position))
        return new{NDIMS, typeof(faces), typeof(vertices)}(faces, vertices)
    end
end

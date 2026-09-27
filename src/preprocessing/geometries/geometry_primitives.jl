# Geometric primitives shared by preprocessing and reconstructed surfaces.
@inline planar_cross(a, b) = a[1] * b[2] - a[2] * b[1]

# Signed tetrahedron contribution of coordinates already shifted by the caller's
# chosen reference point. This preserves the translated reconstruction accumulation
# while `TriangleMesh.volume` can use world coordinates directly.
@inline oriented_tetrahedron_volume(a, b, c) = dot(a, cross(b, c)) / 6

# Both shell containment and the existing winding-number API use the same solid-angle
# denominator. They deliberately keep their separate determinant expressions and sum
# orders: one uses dot/cross, the other computes det([a b c]).
@inline function triangle_solid_angle(a, b, c, a_norm, b_norm, c_norm, numerator)
    denominator = a_norm * b_norm * c_norm + dot(a, b) * c_norm +
                  dot(b, c) * a_norm + dot(c, a) * b_norm
    return 2atan(numerator, denominator)
end

# The seven closest-point Voronoi regions of a triangle, with pseudonormals supplied by
# the caller. Packing and surface reconstruction store triangles/normals differently;
# the kernel does not allocate an intermediate geometry object per query.
# The two historical callers test the ab edge and vertex c in different orders.
# For Voronoi-boundary ties their chosen pseudonormal could differ, so preserve that
# ordering as a compile-time flag rather than changing any tie-breaking behavior.
@inline function triangle_closest_point_and_normal(p, a, b, c, vertex_normals,
                                                   edge_normals, face_normal,
                                                   ::Val{AB_BEFORE_C}) where {AB_BEFORE_C}
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = dot(ab, ap), dot(ac, ap)
    d1 <= 0 && d2 <= 0 && return a, vertex_normals[1]

    bp = p - b
    d3, d4 = dot(ab, bp), dot(ac, bp)
    d3 >= 0 && d4 <= d3 && return b, vertex_normals[2]

    if !AB_BEFORE_C
        cp = p - c
        d5, d6 = dot(ab, cp), dot(ac, cp)
        d6 >= 0 && d5 <= d6 && return c, vertex_normals[3]
    end

    vc = d1 * d4 - d3 * d2
    if vc <= 0 && d1 >= 0 && d3 <= 0
        return a + (d1 / (d1 - d3)) * ab, edge_normals[1]
    end

    if AB_BEFORE_C
        cp = p - c
        d5, d6 = dot(ab, cp), dot(ac, cp)
        d6 >= 0 && d5 <= d6 && return c, vertex_normals[3]
    end

    vb = d5 * d2 - d1 * d6
    if vb <= 0 && d2 >= 0 && d6 <= 0
        return a + (d2 / (d2 - d6)) * ac, edge_normals[3]
    end

    va = d3 * d6 - d5 * d4
    if va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b), edge_normals[2]
    end

    denominator = AB_BEFORE_C ? inv(va + vb + vc) : 1 / (va + vb + vc)
    return a + (vb * denominator) * ab + (vc * denominator) * ac, face_normal
end

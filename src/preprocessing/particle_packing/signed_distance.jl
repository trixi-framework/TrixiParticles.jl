function signed_point_face_distance(p::SVector{3}, boundary, face_index)
    (; face_vertices, normals_vertex, normals_edge, normals_face,
    face_vertices_ids, face_edges_ids) = boundary
    v1 = face_vertices_ids[face_index][1]
    v2 = face_vertices_ids[face_index][2]
    v3 = face_vertices_ids[face_index][3]

    e1 = face_edges_ids[face_index][1]
    e2 = face_edges_ids[face_index][2]
    e3 = face_edges_ids[face_index][3]

    normals_vertices = (normals_vertex[v1], normals_vertex[v2], normals_vertex[v3])

    normals_edges = (normals_edge[e1], normals_edge[e2], normals_edge[e3])

    # Find distance `p` to triangle
    return signed_distance(p, face_vertices[face_index], normals_vertices,
                           normals_edges, normals_face[face_index])
end

function signed_point_face_distance(p::SVector{2}, boundary, face_index)
    (; edge_vertices, normals_vertex, normals_edge) = boundary

    return signed_distance(p, edge_vertices[face_index], normals_vertex[face_index],
                           normals_edge[face_index])
end

# Reference:
# Christer Ericson’s Real-time Collision Detection book
# https://www.r-5.org/files/books/computers/algo-list/realtime-3d/Christer_Ericson-Real-Time_Collision_Detection-EN.pdf
#
# Andreas Bærentzen et al (2002): Generating signed distance fields from triangle meshes
# https://www.researchgate.net/publication/251839082_Generating_Signed_Distance_Fields_From_Triangle_Meshes
#
# Inspired by https://github.com/embree/embree/blob/master/tutorials/common/math/closest_point.h
function signed_distance(p::SVector{3}, triangle_points, normals_vertex, normals_edge, n)
    a = triangle_points[1]
    b = triangle_points[2]
    c = triangle_points[3]

    na = normals_vertex[1]
    nb = normals_vertex[2]
    nc = normals_vertex[3]

    nab = normals_edge[1]
    nbc = normals_edge[2]
    nac = normals_edge[3]

    ab = b - a
    ac = c - a
    ap = p - a

    dot1 = dot(ab, ap)
    dot2 = dot(ac, ap)

    # Region 1: point `a`
    (dot1 <= 0 && dot2 <= 0) && return signbit(dot(ap, na)), dot(ap, ap), na

    bp = p - b

    dot3 = dot(ab, bp)
    dot4 = dot(ac, bp)

    # Region 2: point `b`
    (dot3 >= 0 && dot4 <= dot3) && return signbit(dot(bp, nb)), dot(bp, bp), nb

    cp = p - c

    dot5 = dot(ab, cp)
    dot6 = dot(ac, cp)

    # Region 3: point `c`
    (dot6 >= 0 && dot5 <= dot6) && return signbit(dot(cp, nc)), dot(cp, cp), nc

    vc = dot1 * dot4 - dot3 * dot2

    if vc <= 0 && dot1 >= 0 && dot3 <= 0
        t = dot1 / (dot1 - dot3)

        v = p - (a + t * ab)

        # Region 4: edge `ab`
        return signbit(dot(v, nab)), dot(v, v), nab
    end

    vb = dot5 * dot2 - dot1 * dot6

    if vb <= 0 && dot2 >= 0 && dot6 <= 0
        t = dot2 / (dot2 - dot6)

        v = p - (a + t * ac)

        # Region 5: edge `ac`
        return signbit(dot(v, nac)), dot(v, v), nac
    end

    va = dot3 * dot6 - dot5 * dot4

    if va <= 0 && (dot4 - dot3) >= 0 && (dot5 - dot6) >= 0
        t = (dot4 - dot3) / ((dot4 - dot3) + (dot5 - dot6))

        v = p - (b + t * (c - b))

        # Region 6: edge `bc`
        return signbit(dot(v, nbc)), dot(v, v), nbc
    end

    # Region 0: triangle
    denom = 1 / (va + vb + vc)

    v = vb * denom
    w = vc * denom

    d = p - (a + v * ab + w * ac)

    return signbit(dot(d, n)), dot(d, d), n
end

function signed_distance(p::SVector{2}, edge_points, normals_vertex, n)
    a = edge_points[1]
    b = edge_points[2]

    ab = b - a
    ap = p - a

    na = normals_vertex[1]
    nb = normals_vertex[2]

    dot1 = dot(ab, ab)
    dot2 = dot(ap, ab)

    # Calculate projection of `ap` to `ab`
    proj = dot2 / dot1

    if proj <= 0
        # Closest point is `a`
        return signbit(dot(ap, na)), dot(ap, ap), na
    end

    if proj >= 1
        bp = p - b
        # Closest point is `b`
        return signbit(dot(bp, nb)), dot(bp, bp), nb
    end

    # Closest point is on `ab`
    v = p - (a + proj * ab)

    return signbit(dot(v, n)), dot(v, v), n
end

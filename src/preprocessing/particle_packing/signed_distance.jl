struct SignedDistanceField{NDIMS, ELTYPE}
    positions           :: Vector{SVector{NDIMS, ELTYPE}}
    normals             :: Vector{SVector{NDIMS, ELTYPE}}
    distances           :: Vector{ELTYPE}
    max_signed_distance :: ELTYPE

    function SignedDistanceField(boundary, particle_spacing;
                                 max_signed_distance=4particle_spacing,
                                 neighborhood_search=true, pad=max_signed_distance)
        NDIMS = ndims(boundary)
        ELTYPE = eltype(max_signed_distance)

        if neighborhood_search
            nhs = FaceNeighborhoodSearch{NDIMS}(max_signed_distance)
            initialize!(nhs, boundary)
        else
            nhs = TrivialNeighborhoodSearch{NDIMS}(max_signed_distance, eachface(boundary))
        end

        min_corner = boundary.min_corner .- pad
        max_corner = boundary.max_corner .+ pad
        point_grid = meshgrid(min_corner, max_corner; increment=particle_spacing)

        positions = vec([SVector(position) for position in point_grid])
        normals = fill(SVector(ntuple(dim -> Inf, NDIMS)), length(point_grid))
        distances = fill(Inf, length(point_grid))

        @threaded for point in eachindex(positions)
            point_coords = positions[point]

            for face in eachneighbor(point_coords, nhs)
                # `sdf = (sign, distance, normal)`
                sdf = signed_point_face_distance(point_coords, boundary, face)

                if sdf[2] <= (max_signed_distance)^2 && sdf[2] < distances[point]^2
                    distances[point] = sdf[1] ? -sqrt(sdf[2]) : sqrt(sdf[2])
                    normals[point] = sdf[3]
                end
            end
        end

        reject_indices = distances .== Inf

        deleteat!(distances, reject_indices)
        deleteat!(normals, reject_indices)
        deleteat!(positions, reject_indices)

        return new{NDIMS, ELTYPE}(positions, normals, distances, max_signed_distance)
    end
end

@inline Base.ndims(::SignedDistanceField{NDIMS}) where {NDIMS} = NDIMS

function Base.show(io::IO, system::SignedDistanceField)
    @nospecialize system # reduce precompilation time

    print(io, "SignedDistanceField{", ndims(system), "}()")
end

function Base.show(io::IO, ::MIME"text/plain", system::SignedDistanceField)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "SignedDistanceField{$(ndims(system))}")
        summary_line(io, "#particles", length(system.distances))
        summary_line(io, "max signed distance", system.max_signed_distance)
        summary_footer(io)
    end
end

function signed_point_face_distance(p::SVector{2}, boundary, edge_index)
    (; edge_vertices, normals_vertex, normals_edge) = boundary

    n = normals_edge[edge_index]

    a = edge_vertices[edge_index][1]
    b = edge_vertices[edge_index][2]

    ab = b - a
    ap = p - a

    na = normals_vertex[edge_index][1]
    nb = normals_vertex[edge_index][2]

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

# Reference:
# Christer Ericson’s Real-time Collision Detection book
# https://www.r-5.org/files/books/computers/algo-list/realtime-3d/Christer_Ericson-Real-Time_Collision_Detection-EN.pdf
#
# Andreas Bærentzen et al (2002): Generating signed distance fields from triangle meshes
# https://www.researchgate.net/publication/251839082_Generating_Signed_Distance_Fields_From_Triangle_Meshes
#
# Inspired by https://github.com/embree/embree/blob/master/tutorials/common/math/closest_point.h
function signed_point_face_distance(p::SVector{3}, boundary, face_index)
    (; face_vertices, face_vertices_ids, normals_edge,
    face_edges_ids, normals_face, normals_vertex) = boundary

    a = face_vertices[face_index][1]
    b = face_vertices[face_index][2]
    c = face_vertices[face_index][3]

    n = normals_face[face_index]

    v1 = face_vertices_ids[face_index][1]
    v2 = face_vertices_ids[face_index][2]
    v3 = face_vertices_ids[face_index][3]

    e1 = face_edges_ids[face_index][1]
    e2 = face_edges_ids[face_index][2]
    e3 = face_edges_ids[face_index][3]

    na = normals_vertex[v1]
    nb = normals_vertex[v2]
    nc = normals_vertex[v3]

    nab = normals_edge[e1]
    nbc = normals_edge[e2]
    nac = normals_edge[e3]

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

    u = vb * denom
    w = vc * denom

    d = p - (a + u * ab + w * ac)

    return signbit(dot(d, n)), dot(d, d), n
end

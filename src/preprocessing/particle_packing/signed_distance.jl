"""
    SignedDistanceField(geometry, particle_spacing;
                        points=nothing,
                        max_signed_distance=4 * particle_spacing,
                        use_for_boundary_packing=false)

Generate particles along a surface of a complex geometry storing the signed distances and normals
to this surface.

# Arguments
- `geometry`: Geometry returned by [`load_geometry`](@ref).
- `particle_spacing`: Spacing between the particles.

# Keywords
- `max_signed_distance`:      Maximum signed distance to be stored. That is, only particles with a
                              distance of `abs(max_signed_distance)` to the surface of the shape
                              will be sampled.
- `points`:                   Points on which the signed distance is computed.
                              When set to `nothing` (default), the bounding box of the shape will be
                              sampled with a uniform grid of points.
- `use_for_boundary_packing`: Set to `true` if [`SignedDistanceField`] is used to pack
                              a boundary [`ParticlePackingSystem`](@ref).
                              Use the default of `false` when packing without a boundary.
"""
struct SignedDistanceField{ELTYPE, P, D}
    positions           :: P
    normals             :: P
    distances           :: D
    max_signed_distance :: ELTYPE
    boundary_packing    :: Bool
    particle_spacing    :: ELTYPE
end

function SignedDistanceField(geometry, particle_spacing;
                             points=nothing, neighborhood_search=true,
                             max_signed_distance=4 * particle_spacing,
                             use_for_boundary_packing=false)
    NDIMS = ndims(geometry)
    ELTYPE = eltype(max_signed_distance)

    sdf_factor = use_for_boundary_packing ? 2 : 1

    search_radius = sdf_factor * max_signed_distance

    if neighborhood_search
        nhs = FaceNeighborhoodSearch{NDIMS}(; search_radius)
    else
        nhs = TrivialNeighborhoodSearch{NDIMS}(eachpoint=eachface(geometry))
    end

    initialize!(nhs, geometry)

    if isnothing(points)
        min_corner = geometry.min_corner .- search_radius
        max_corner = geometry.max_corner .+ search_radius

        n_particles_per_dimension = Tuple(ceil.(Int,
                                                (max_corner .- min_corner) ./
                                                particle_spacing))

        grid = rectangular_shape_coords(particle_spacing, n_particles_per_dimension,
                                        min_corner; place_on_shell=true)

        points = reinterpret(reshape, SVector{NDIMS, ELTYPE}, grid)
    end

    positions = copy(points)

    # This gives a performance boost for large geometries
    delete_positions_in_empty_cells!(positions, nhs)

    normals = fill(SVector(ntuple(dim -> Inf, NDIMS)), length(positions))
    distances = fill(Inf, length(positions))

    calculate_signed_distances!(positions, distances, normals,
                                geometry, sdf_factor, max_signed_distance, nhs)

    return SignedDistanceField(positions, normals, distances, max_signed_distance,
                               use_for_boundary_packing, particle_spacing)
end

function Base.show(io::IO, system::SignedDistanceField)
    @nospecialize system # reduce precompilation time

    print(io, "SignedDistanceField()")
end

function Base.show(io::IO, ::MIME"text/plain", system::SignedDistanceField)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "SignedDistanceField")
        summary_line(io, "#particles", length(system.distances))
        summary_line(io, "max signed distance", system.max_signed_distance)
        summary_footer(io)
    end
end

function trixi2vtk(signed_distance_field::SignedDistanceField;
                   filename="signed_distance_field", output_directory="out")
    (; positions, distances, normals) = signed_distance_field
    positions = stack(signed_distance_field.positions)

    trixi2vtk(positions, signed_distances=distances, normals=normals,
              filename=filename, output_directory=output_directory)
end

delete_positions_in_empty_cells!(positions, nhs::TrivialNeighborhoodSearch) = positions

function delete_positions_in_empty_cells!(positions, nhs::FaceNeighborhoodSearch)
    delete_positions = fill(false, length(positions))

    @threaded default_backend(positions) for point in eachindex(positions)
        if isempty(eachneighbor(positions[point], nhs))
            delete_positions[point] = true
        end
    end

    deleteat!(positions, delete_positions)

    return positions
end

function calculate_signed_distances!(positions, distances, normals,
                                     boundary, sdf_factor, max_signed_distance, nhs)
    @threaded default_backend(positions) for point in eachindex(positions)
        point_coords = positions[point]

        for face in eachneighbor(point_coords, nhs)
            sign_bit, distance,
            normal = signed_point_face_distance(point_coords, boundary,
                                                face)

            if distance < distances[point]^2
                # Found a face closer than the previous closest face
                distances[point] = sign_bit ? -sqrt(distance) : sqrt(distance)
                normals[point] = normal
            end
        end
    end

    # Keep "larger" signed distance field outside `boundary` to guarantee
    # compact support for boundary particles.
    keep = -max_signed_distance .< distances .< sdf_factor * max_signed_distance
    delete_indices = .!keep

    deleteat!(distances, delete_indices)
    deleteat!(normals, delete_indices)
    deleteat!(positions, delete_indices)

    return positions
end

function signed_point_face_distance(p::SVector{2}, boundary, edge_index)
    (; edge_vertices, vertex_normals, edge_normals) = boundary

    n = edge_normals[edge_index]

    a = edge_vertices[edge_index][1]
    b = edge_vertices[edge_index][2]

    ab = b - a
    ap = p - a

    na = vertex_normals[edge_index][1]
    nb = vertex_normals[edge_index][2]

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
    (; face_vertices, face_vertices_ids, edge_normals,
     face_edges_ids, face_normals, vertex_normals) = boundary

    a = face_vertices[face_index][1]
    b = face_vertices[face_index][2]
    c = face_vertices[face_index][3]

    n = face_normals[face_index]

    v1 = face_vertices_ids[face_index][1]
    v2 = face_vertices_ids[face_index][2]
    v3 = face_vertices_ids[face_index][3]

    e1 = face_edges_ids[face_index][1]
    e2 = face_edges_ids[face_index][2]
    e3 = face_edges_ids[face_index][3]

    na = vertex_normals[v1]
    nb = vertex_normals[v2]
    nc = vertex_normals[v3]

    nab = edge_normals[e1]
    nbc = edge_normals[e2]
    nac = edge_normals[e3]

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

struct FaceNeighborhoodSearch{NDIMS, ELTYPE, NP}
    hashtable     :: Dict{NTuple{NDIMS, Int}, Vector{Int}}
    nhs_particles :: NP
    cell_size     :: NTuple{NDIMS, ELTYPE}
    spanning_set  :: Vector{Vector{SVector{NDIMS, ELTYPE}}}

    function FaceNeighborhoodSearch(nhs_particles)
        NDIMS = ndims(nhs_particles)
        ELTYPE = eltype(nhs_particles.search_radius)

        hashtable = Dict{NTuple{NDIMS, Int}, Vector{Int}}()

        radi = [nhs_particles.search_radius for i in 1:NDIMS]
        vecs = reinterpret(reshape, SVector{NDIMS, ELTYPE}, I(NDIMS) .* radi')

        spanning_set = [[fill(0.0, SVector{NDIMS}), vecs[i]] for i in 1:NDIMS]

        new{NDIMS, ELTYPE,
            typeof(nhs_particles)}(empty!(hashtable), nhs_particles,
                                   nhs_particles.cell_size, spanning_set)
    end
end

function initialize!(neighborhood_search::FaceNeighborhoodSearch{2}, boundary)
    (; hashtable, nhs_particles) = neighborhood_search
    (; edge_vertices) = boundary

    empty!(hashtable)

    for cell in keys(nhs_particles.hashtable)
        x, y = cell
        neighboring_cells = ((x + i, y + j) for i in -1:1, j in -1:1)

        for cell_neighbor in neighboring_cells
            for (face_id, face) in enumerate(edge_vertices)
                if cell_intersection(face, cell_neighbor, neighborhood_search)
                    if haskey(hashtable, cell_neighbor) &&
                       !(face_id in hashtable[cell_neighbor])

                        # Add particle to corresponding cell
                        append!(hashtable[cell_neighbor], face_id)

                    else
                        # Create cell
                        hashtable[cell_neighbor] = [face_id]
                    end
                end
            end
        end
    end

    return neighborhood_search
end

# 2D
@inline function eachneighborface(coords, nhs_faces::FaceNeighborhoodSearch{2})
    cell = TrixiParticles.cell_coords(coords, nhs_faces.nhs_particles)
    x, y = cell
    # Generator of all neighboring cells to consider
    neighboring_cells = ((x + i, y + j) for i in -1:1, j in -1:1)

    # Merge all lists of particles in the neighboring cells into one iterator
    # TODO make a non-allocating version
    return unique(Iterators.flatten(faces_in_cell(cell, nhs_faces)
                                    for cell in neighboring_cells))
end

@inline eachneighborface(coords, nhs_faces::TrivialNeighborhoodSearch) = nhs_faces.eachparticle

@inline function faces_in_cell(cell_index, neighborhood_search)
    (; hashtable, nhs_particles) = neighborhood_search
    (; empty_vector) = nhs_particles

    # Return an empty vector when `cell_index` is not a key of `hashtable` and
    # reuse the empty vector to avoid allocations
    return get(hashtable, cell_index, empty_vector)
end

function cell_intersection(edge, cell, nhs)
    (; spanning_set, cell_size) = nhs

    min_corner = SVector(cell .* cell_size...)

    v1 = spanning_set[1][2]
    v2 = spanning_set[2][2]

    position_1 = edge[1] - min_corner
    position_2 = edge[2] - min_corner

    # Check if any edge point is in cell
    point_in_cell(position_1, v1, v2) && return true
    point_in_cell(position_2, v1, v2) && return true

    cell_edge_1 = [spanning_set[1][i] + min_corner for i in 1:2]
    cell_edge_2 = [spanning_set[2][i] + min_corner for i in 1:2]

    # Check edge intersection with cell edges
    edge_intersection(edge, cell_edge_1) && return true
    edge_intersection(edge, cell_edge_2) && return true

    return false
end

function edge_intersection(edge_1, edge_2)
    a = edge_1[1]
    b = edge_1[2]

    c = edge_2[1]
    d = edge_2[2]

    # Helper function
    det2D(ab, cd) = ab[1] * cd[2] - ab[2] * cd[1]

    v1 = b - a
    v2 = d - a
    v3 = c - a

    det_1 = det2D(v1, v2)
    det_2 = det2D(v1, v3)

    if det_1 * det_2 <= 0
        v4 = d - c
        v5 = a - c
        v6 = b - c

        det_3 = det2D(v4, v5)
        det_4 = det2D(v4, v6)

        if det_3 * det_4 <= 0
            return true
        end
    end

    return false
end

@inline function point_in_cell(p, v1, v2)
    !(0 <= dot(p, v1) <= dot(v1, v1)) && return false
    !(0 <= dot(p, v2) <= dot(v2, v2)) && return false

    return true
end

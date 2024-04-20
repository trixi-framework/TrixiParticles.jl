struct FaceNeighborhoodSearch{NDIMS, ELTYPE, NP}
    hashtable     :: Dict{NTuple{NDIMS, Int}, Vector{Int}}
    nhs_particles :: NP
    cell_size     :: NTuple{NDIMS, ELTYPE}

    function FaceNeighborhoodSearch(nhs_particles)
        NDIMS = ndims(nhs_particles)
        ELTYPE = eltype(nhs_particles.search_radius)

        hashtable = Dict{NTuple{NDIMS, Int}, Vector{Int}}()

        new{NDIMS, ELTYPE,
            typeof(nhs_particles)}(empty!(hashtable), nhs_particles,
                                   nhs_particles.cell_size)
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

# No nhs
@inline eachneighborface(coords, nhs_faces::TrivialNeighborhoodSearch) = nhs_faces.eachparticle

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

@inline function faces_in_cell(cell_index, neighborhood_search)
    (; hashtable, nhs_particles) = neighborhood_search
    (; empty_vector) = nhs_particles

    # Return an empty vector when `cell_index` is not a key of `hashtable` and
    # reuse the empty vector to avoid allocations
    return get(hashtable, cell_index, empty_vector)
end

function cell_intersection(edge, cell, nhs)
    (; cell_size) = nhs

    # Check if any edge point is in cell
    cell == cell_coords(edge[1], nhs.nhs_particles) && return true
    cell == cell_coords(edge[2], nhs.nhs_particles) && return true

    min_corner = SVector(cell .* cell_size...)
    max_corner = min_corner + SVector(cell_size...)

    ray_direction = edge[2] - edge[1]

    return ray_intersection(min_corner, max_corner, edge[1], ray_direction)
end

function ray_intersection(min_corner, max_corner, ray_origin, ray_direction)
    inv_dirx = 1 / ray_direction[1]
    inv_diry = 1 / ray_direction[2]

    tx1 = (min_corner[1] - ray_origin[1]) * inv_dirx
    tx2 = (max_corner[1] - ray_origin[1]) * inv_dirx

    tmin = min(tx1, tx2)
    tmax = max(tx1, tx2)

    ty1 = (min_corner[2] - ray_origin[2]) * inv_diry
    ty2 = (max_corner[2] - ray_origin[2]) * inv_diry

    tmin = max(tmin, min(ty1, ty2))
    tmax = min(tmax, max(ty1, ty2))

    return tmin <= tmax
end

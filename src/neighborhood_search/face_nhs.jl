struct FaceNeighborhoodSearch{NDIMS, ELTYPE, NP}
    hashtable         :: Dict{NTuple{NDIMS, Int}, Vector{Int}}
    nhs_particles     :: NP
    cell_size         :: NTuple{NDIMS, ELTYPE}
    neighbor_iterator :: Dict{NTuple{NDIMS, Int}, Vector{Int}}

    function FaceNeighborhoodSearch(nhs_particles)
        NDIMS = ndims(nhs_particles)
        ELTYPE = eltype(nhs_particles.search_radius)

        hashtable = Dict{NTuple{NDIMS, Int}, Vector{Int}}()
        neighbor_iterator = Dict{NTuple{NDIMS, Int}, Vector{Int}}()

        new{NDIMS, ELTYPE,
            typeof(nhs_particles)}(empty!(hashtable), nhs_particles,
                                   nhs_particles.cell_size, neighbor_iterator)
    end
end

function initialize!(neighborhood_search::FaceNeighborhoodSearch, boundary)
    (; hashtable, nhs_particles, neighbor_iterator) = neighborhood_search

    empty!(hashtable)

    for cell in keys(nhs_particles.hashtable)
        for cell_neighbor in neighboring_cells(cell)
            for face in eachface(boundary)
                if cell_intersection(boundary, face, cell_neighbor, neighborhood_search)
                    if haskey(hashtable, cell_neighbor) &&
                       !(face in hashtable[cell_neighbor])

                        # Add particle to corresponding cell
                        append!(hashtable[cell_neighbor], face)

                    else
                        # Create cell
                        hashtable[cell_neighbor] = [face]
                    end
                end
            end
        end
    end

    for cell in keys(hashtable)
        neighbor_iterator[cell] = unique(Iterators.flatten(faces_in_cell(cell,
                                                                         neighborhood_search)
                                                           for cell in neighboring_cells(cell)))
    end

    return neighborhood_search
end

# No nhs
@inline eachneighborface(coords, nhs_faces::TrivialNeighborhoodSearch) = nhs_faces.eachparticle

@inline function eachneighborface(coords, nhs_faces::FaceNeighborhoodSearch)
    cell = TrixiParticles.cell_coords(coords, nhs_faces.nhs_particles)

    haskey(nhs_faces.neighbor_iterator, cell) && return nhs_faces.neighbor_iterator[cell]

    return nhs_faces.nhs_particles.empty_vector
end

@inline function faces_in_cell(cell_index, neighborhood_search)
    (; hashtable, nhs_particles) = neighborhood_search
    (; empty_vector) = nhs_particles

    # Return an empty vector when `cell_index` is not a key of `hashtable` and
    # reuse the empty vector to avoid allocations
    return get(hashtable, cell_index, empty_vector)
end

function cell_intersection(boundary, edge_index, cell, nhs::FaceNeighborhoodSearch{2})
    (; edge_vertices) = boundary
    (; cell_size) = nhs

    edge = edge_vertices[edge_index]

    # Check if any edge point is in cell
    cell == cell_coords(edge[1], nhs.nhs_particles) && return true
    cell == cell_coords(edge[2], nhs.nhs_particles) && return true

    # Check if line segment intersects cell
    min_corner = SVector(cell .* cell_size...)
    max_corner = min_corner + SVector(cell_size...)

    ray_direction = edge[2] - edge[1]

    return ray_intersection(min_corner, max_corner, edge[1], ray_direction)
end

function cell_intersection(boundary, face_index, cell, nhs::FaceNeighborhoodSearch{3})
    (; face_vertices, normals_face) = boundary
    (; cell_size) = nhs

    vertices = face_vertices[face_index]

    # Check if any vertex is in cell
    cell == cell_coords(vertices[1], nhs.nhs_particles) && return true
    cell == cell_coords(vertices[2], nhs.nhs_particles) && return true
    cell == cell_coords(vertices[3], nhs.nhs_particles) && return true

    # Check if line segment intersects cell
    min_corner = SVector(cell .* cell_size...)
    cell_size_ = SVector(cell_size...)
    max_corner = min_corner + cell_size_

    ray_direction1 = vertices[2] - vertices[1]
    ray_intersection(min_corner, max_corner, vertices[1], ray_direction1) && return true

    ray_direction2 = vertices[2] - vertices[3]
    ray_intersection(min_corner, max_corner, vertices[1], ray_direction2) && return true

    ray_direction3 = vertices[3] - vertices[1]
    ray_intersection(min_corner, max_corner, vertices[1], ray_direction3) && return true

    n = normals_face[face_index]

    # Check if triangle plane intersects cell
    return triangle_plane_intersection(vertices, min_corner, max_corner, cell_size_, n)
end

# See https://tavianator.com/2022/ray_box_boundary.html
function ray_intersection(min_corner, max_corner, ray_origin, ray_direction)
    NDIMS = length(ray_origin)

    inv_dir = SVector(ntuple(@inline(dim->1 / ray_direction[dim]), NDIMS))

    tmin = zero(eltype(ray_direction))
    tmax = Inf
    for dim in 1:NDIMS
        t1 = (min_corner[dim] - ray_origin[dim]) * inv_dir[dim]
        t2 = (max_corner[dim] - ray_origin[dim]) * inv_dir[dim]

        tmin = min(max(t1, tmin), max(t2, tmin))
        tmax = max(min(t1, tmax), min(t2, tmax))
    end

    return tmin <= tmax
end

function ray_intersection(min_corner, max_corner, ray_origin, ray_direction::SVector{2})
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

# TODO: Fix false-negative problem (use signed distance function?)
function triangle_plane_intersection(vertices, min_corner, max_corner, cell_size, n)
    v = vertices[1]

    dirx = SVector(cell_size[1], zero(eltype(v)), zero(eltype(v)))
    diry = SVector(zero(eltype(v)), cell_size[2], zero(eltype(v)))
    dirz = SVector(zero(eltype(v)), zero(eltype(v)), cell_size[3])

    cell_vertex_1 = min_corner
    cell_vertex_2 = min_corner + dirx

    pos_diff_1 = cell_vertex_1 - v
    pos_diff_2 = cell_vertex_2 - v

    dot1 = sign(dot(pos_diff_1, n))
    dot2 = sign(dot(pos_diff_2, n))
    !(dot1 == dot2) && return true

    cell_vertex_3 = min_corner + diry
    pos_diff_3 = cell_vertex_3 - v

    dot3 = sign(dot(pos_diff_3, n))
    !(dot2 == dot3) && return true

    cell_vertex_4 = min_corner + dirz
    pos_diff_4 = cell_vertex_4 - v

    dot4 = sign(dot(pos_diff_4, n))
    !(dot3 == dot4) && return true

    cell_vertex_5 = max_corner
    pos_diff_5 = cell_vertex_5 - v

    dot5 = sign(dot(pos_diff_5, n))
    !(dot4 == dot5) && return true

    cell_vertex_6 = max_corner - dirx
    pos_diff_6 = cell_vertex_6 - v

    dot6 = sign(dot(pos_diff_6, n))
    !(dot5 == dot6) && return true

    cell_vertex_7 = max_corner - diry
    pos_diff_7 = cell_vertex_7 - v

    dot7 = sign(dot(pos_diff_7, n))
    !(dot6 == dot7) && return true

    cell_vertex_8 = max_corner - dirz
    pos_diff_8 = cell_vertex_8 - v

    dot8 = sign(dot(pos_diff_8, n))
    !(dot7 == dot8) && return true

    # All edge vertices are on one side of the plane
    return false
end

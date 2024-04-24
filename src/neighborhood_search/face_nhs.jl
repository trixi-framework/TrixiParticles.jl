struct FaceNeighborhoodSearch{NDIMS, ELTYPE}
    hashtable_face    :: Dict{NTuple{NDIMS, Int}, Vector{Int}}
    hashtable_sdf     :: Dict{NTuple{NDIMS, Int}, Vector{Int}}
    cell_size         :: NTuple{NDIMS, ELTYPE}
    neighbor_iterator :: Dict{NTuple{NDIMS, Int}, Vector{Int}}

    function FaceNeighborhoodSearch{NDIMS}(search_radius; subdivision=1) where {NDIMS}
        ELTYPE = eltype(search_radius)

        hashtable_face = Dict{NTuple{NDIMS, Int}, Vector{Int}}()
        hashtable_sdf = Dict{NTuple{NDIMS, Int}, Vector{Int}}()
        neighbor_iterator = Dict{NTuple{NDIMS, Int}, Vector{Int}}()

        cell_size = ntuple(dim -> search_radius, NDIMS) ./ subdivision

        new{NDIMS, ELTYPE}(hashtable_face, hashtable_sdf,
                           cell_size, neighbor_iterator)
    end
end

function initialize!(neighborhood_search::FaceNeighborhoodSearch, system)
    (; hashtable_face, neighbor_iterator, cell_size) = neighborhood_search
    (; boundary, initial_condition) = system

    empty!(hashtable_face)

    for face in eachface(boundary)
        for cell in cell_grid(face, boundary, neighborhood_search)
            if cell_intersection(boundary, face, cell, neighborhood_search)
                if haskey(hashtable_face, cell) && !(face in hashtable_face[cell])

                    # Add particle to corresponding cell
                    append!(hashtable_face[cell], face)

                else
                    # Create cell
                    hashtable_face[cell] = [face]
                end
            end
        end
    end

    min_corner = (minimum(initial_condition.coordinates[i, :]) for i in 1:ndims(system))
    max_corner = (maximum(initial_condition.coordinates[i, :]) for i in 1:ndims(system))

    # Padding
    cell_min = Tuple(floor_to_int.(min_corner ./ cell_size)) .- 2
    cell_max = Tuple(floor_to_int.(max_corner ./ cell_size)) .+ 2

    for cell_runner in meshgrid(cell_min, cell_max)
        if any([haskey(hashtable_face, cell_) for cell_ in neighboring_cells(cell_runner)])
            for cell in neighboring_cells(cell_runner)
                neighbor_iterator[cell] = unique(Iterators.flatten(faces_in_cell(neighbor,
                                                                                 neighborhood_search)
                                                                   for neighbor in neighboring_cells(cell)))
            end
        end
    end

    system.precalculate_sdf && calculate_signed_distance!(system)

    return neighborhood_search
end

@inline function meshgrid(min_corner, max_corner; increment=1)
    min_ = collect(min_corner)
    max_ = collect(max_corner)

    ranges = ntuple(dim -> (min_[dim]:increment:max_[dim]), length(min_corner))

    return Iterators.product(ranges...)
end

@inline function cell_grid(edge, shape, neighborhood_search::FaceNeighborhoodSearch{2})
    (; cell_size) = neighborhood_search

    v1 = shape.edge_vertices[edge][1]
    v2 = shape.edge_vertices[edge][2]

    cell1 = Tuple(floor_to_int.(v1 ./ cell_size))
    cell2 = Tuple(floor_to_int.(v2 ./ cell_size))

    # Create cell grid
    mins = min.(cell1, cell2)
    maxs = max.(cell1, cell2)

    return meshgrid(mins, maxs)
end

@inline function cell_grid(face, shape, neighborhood_search::FaceNeighborhoodSearch{3})
    (; cell_size) = neighborhood_search

    v1 = shape.face_vertices[face][1]
    v2 = shape.face_vertices[face][2]
    v3 = shape.face_vertices[face][3]

    cell1 = Tuple(floor_to_int.(v1 ./ cell_size))
    cell2 = Tuple(floor_to_int.(v2 ./ cell_size))
    cell3 = Tuple(floor_to_int.(v3 ./ cell_size))

    # Create cell grid
    mins = min.(cell1, cell2, cell3)
    maxs = max.(cell1, cell2, cell3)

    return meshgrid(mins, maxs)
end

# No nhs
@inline eachneighborface(coords, nhs_faces::TrivialNeighborhoodSearch) = nhs_faces.eachparticle

@inline function eachneighborface(coords, nhs_faces::FaceNeighborhoodSearch)
    cell = TrixiParticles.cell_coords(coords, nothing, nhs_faces.cell_size)

    haskey(nhs_faces.neighbor_iterator, cell) && return nhs_faces.neighbor_iterator[cell]

    return Int[]
end

@inline function eachneighbor_distances(coords, nhs_faces)
    cell = TrixiParticles.cell_coords(coords, nothing, nhs_faces.cell_size)

    # Merge all lists of particles in the neighboring cells into one iterator
    return Iterators.flatten(points_in_cell(cell, nhs_faces)
                             for cell in neighboring_cells(cell))
end

@inline function points_in_cell(cell_index, neighborhood_search)
    (; hashtable_sdf) = neighborhood_search

    # Return an empty vector when `cell_index` is not a key of `hashtable` and
    # reuse the empty vector to avoid allocations
    return get(hashtable_sdf, cell_index, Int[])
end

@inline function faces_in_cell(cell_index, neighborhood_search)
    (; hashtable_face) = neighborhood_search

    # Return an empty vector when `cell_index` is not a key of `hashtable` and
    # reuse the empty vector to avoid allocations
    return get(hashtable_face, cell_index, Int[])
end

function cell_intersection(boundary, edge_index, cell, nhs::FaceNeighborhoodSearch{2})
    (; edge_vertices) = boundary
    (; cell_size) = nhs

    edge = edge_vertices[edge_index]

    # Check if any edge point is in cell
    cell == cell_coords(edge[1], nothing, nhs.cell_size) && return true
    cell == cell_coords(edge[2], nothing, nhs.cell_size) && return true

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
    cell == cell_coords(vertices[1], nothing, nhs.cell_size) && return true
    cell == cell_coords(vertices[2], nothing, nhs.cell_size) && return true
    cell == cell_coords(vertices[3], nothing, nhs.cell_size) && return true

    # Check if line segment intersects cell
    min_corner = SVector(cell .* cell_size...)
    max_corner = min_corner + SVector(cell_size...)

    ray_direction1 = vertices[2] - vertices[1]
    ray_intersection(min_corner, max_corner, vertices[1], ray_direction1) && return true

    ray_direction2 = vertices[2] - vertices[3]
    ray_intersection(min_corner, max_corner, vertices[1], ray_direction2) && return true

    ray_direction3 = vertices[3] - vertices[1]
    ray_intersection(min_corner, max_corner, vertices[1], ray_direction3) && return true

    n = normals_face[face_index]

    # Check if triangle plane intersects cell
    return triangle_plane_intersection(vertices, min_corner, max_corner, cell_size, n)
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

# TODO: Realy needs test. Is there a false-negative problem? (use signed distance function?)
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

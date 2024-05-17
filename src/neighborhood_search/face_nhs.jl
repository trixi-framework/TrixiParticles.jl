struct FaceNeighborhoodSearch{NDIMS, ELTYPE, PB}
    hashtable         :: Dict{NTuple{NDIMS, Int}, Vector{Int}}
    cell_size         :: NTuple{NDIMS, ELTYPE}
    neighbor_iterator :: Dict{NTuple{NDIMS, Int}, Vector{Int}}
    empty_vector      :: Vector{Int} # Just an empty vector (used in `eachneighbor`)
    periodic_box      :: PB # TODO
    n_cells           :: NTuple{NDIMS, Int}

    function FaceNeighborhoodSearch{NDIMS}(search_radius; subdivision=1) where {NDIMS}
        ELTYPE = eltype(search_radius)

        hashtable = Dict{NTuple{NDIMS, Int}, Vector{Int}}()
        neighbor_iterator = Dict{NTuple{NDIMS, Int}, Vector{Int}}()

        cell_size = ntuple(dim -> search_radius, NDIMS) ./ subdivision
        empty_vector = Int[]

        n_cells = ntuple(_ -> -1, Val(NDIMS))

        new{NDIMS, ELTYPE, Nothing}(hashtable, cell_size, neighbor_iterator, empty_vector,
                                    nothing, n_cells)
    end
end

@inline function eachneighbor(coords, neighborhood_search::FaceNeighborhoodSearch)
    (; neighbor_iterator, empty_vector) = neighborhood_search
    cell = cell_coords(coords, neighborhood_search)

    haskey(neighbor_iterator, cell) && return neighbor_iterator[cell]

    return empty_vector
end

faces_in_cell(cell, neighborhood_search) = particles_in_cell(cell, neighborhood_search)

function initialize!(neighborhood_search::FaceNeighborhoodSearch, mesh)
    (; hashtable, neighbor_iterator) = neighborhood_search

    empty!(hashtable)

    # Fill cells with intersecting faces
    for face in eachface(mesh)

        # Check if any face intersects a cell in the face-embedding cell grid
        for cell in cell_grid(face, mesh, neighborhood_search)
            if cell_intersection(face, mesh, cell, neighborhood_search)
                if haskey(hashtable, cell) && !(face in hashtable[cell])

                    # Add face to corresponding cell
                    append!(hashtable[cell], face)
                else
                    # Create cell
                    hashtable[cell] = [face]
                end
            end
        end
    end

    for cell in keys(hashtable)
        for neighbor in neighboring_cells(cell)
            # Merge all lists of faces in the neighboring cells into one iterator
            neighbor_iterator[neighbor] = unique(Iterators.flatten(faces_in_cell(neighbor_,
                                                                                 neighborhood_search)
                                                                   for neighbor_ in neighboring_cells(neighbor)))
        end
    end
end

function cell_intersection(edge, shape, cell,
                           neighborhood_search::FaceNeighborhoodSearch{2})
    (; cell_size) = neighborhood_search

    v1, v2 = face_vertices(edge, shape)

    # Check if one of the vertices is inside cell
    cell == cell_coords(v1, neighborhood_search) && return true
    cell == cell_coords(v2, neighborhood_search) && return true

    # Check if line segment intersects cell
    min_corner = SVector(cell .* cell_size...)
    max_corner = min_corner + SVector(cell_size...)

    ray_direction = v2 - v1

    return ray_intersection(min_corner, max_corner, v1, ray_direction)
end

function cell_intersection(triangle, shape, cell,
                           neighborhood_search::FaceNeighborhoodSearch{3})
    (; cell_size) = neighborhood_search

    v1, v2, v3 = face_vertices(triangle, shape)

    # Check if one of the vertices is inside cell
    cell == cell_coords(v1, neighborhood_search) && return true
    cell == cell_coords(v2, neighborhood_search) && return true
    cell == cell_coords(v3, neighborhood_search) && return true

    # Check if line segment intersects cell
    min_corner = SVector(cell .* cell_size...)
    max_corner = min_corner + SVector(cell_size...)

    ray_direction1 = v2 - v1
    ray_intersection(min_corner, max_corner, v1, ray_direction1) && return true

    ray_direction2 = v2 - v3
    ray_intersection(min_corner, max_corner, v1, ray_direction2) && return true

    ray_direction3 = v3 - v1
    ray_intersection(min_corner, max_corner, v1, ray_direction3) && return true

    normal = face_normal(triangle, shape)

    # Check if triangle plane intersects cell (for very large triangles)
    return triangle_plane_intersection(v1, normal, min_corner, max_corner, cell_size)
end

# See https://tavianator.com/2022/ray_box_boundary.html
function ray_intersection(min_corner, max_corner, ray_origin, ray_direction)
    NDIMS = length(ray_origin)

    inv_dir = SVector(ntuple(@inline(dim->1 / ray_direction[dim]), NDIMS))

    tmin = zero(eltype(ray_direction))
    tmax = Inf
    @inbounds for dim in 1:NDIMS
        t1 = (min_corner[dim] - ray_origin[dim]) * inv_dir[dim]
        t2 = (max_corner[dim] - ray_origin[dim]) * inv_dir[dim]

        tmin = min(max(t1, tmin), max(t2, tmin))
        tmax = max(min(t1, tmax), min(t2, tmax))
    end

    return tmin <= tmax
end

# Check if each cell vertex is located on the same side of the plane.
# Otherwise the plane intersects the cell.
function triangle_plane_intersection(point, plane_normal, min_corner, max_corner, cell_size)
    dirx = SVector(cell_size[1], zero(eltype(point)), zero(eltype(point)))
    diry = SVector(zero(eltype(point)), cell_size[2], zero(eltype(point)))
    dirz = SVector(zero(eltype(point)), zero(eltype(point)), cell_size[3])

    cell_vertex_1 = min_corner
    cell_vertex_2 = min_corner + dirx

    pos_diff_1 = cell_vertex_1 - point
    pos_diff_2 = cell_vertex_2 - point

    # Corners: bottom north-west and bottom north-east
    dot1 = sign(dot(pos_diff_1, plane_normal))
    dot2 = sign(dot(pos_diff_2, plane_normal))
    !(dot1 == dot2) && return true

    cell_vertex_3 = min_corner + diry
    pos_diff_3 = cell_vertex_3 - point

    # Corners: bottom north-east and top north-west
    dot3 = sign(dot(pos_diff_3, plane_normal))
    !(dot2 == dot3) && return true

    cell_vertex_4 = min_corner + dirz
    pos_diff_4 = cell_vertex_4 - point

    # Corners: top norht-west and bottom south-west
    dot4 = sign(dot(pos_diff_4, plane_normal))
    !(dot3 == dot4) && return true

    cell_vertex_5 = max_corner
    pos_diff_5 = cell_vertex_5 - point

    # Corners: bottom south-west and top south-east
    dot5 = sign(dot(pos_diff_5, plane_normal))
    !(dot4 == dot5) && return true

    cell_vertex_6 = max_corner - dirx
    pos_diff_6 = cell_vertex_6 - point

    # Corners: top south-east and top south-west
    dot6 = sign(dot(pos_diff_6, plane_normal))
    !(dot5 == dot6) && return true

    cell_vertex_7 = max_corner - diry
    pos_diff_7 = cell_vertex_7 - point

    # Corners: top south-west and bottom south-east
    dot7 = sign(dot(pos_diff_7, plane_normal))
    !(dot6 == dot7) && return true

    cell_vertex_8 = max_corner - dirz
    pos_diff_8 = cell_vertex_8 - point

    # Corners: bottom south-east and top north-east
    dot8 = sign(dot(pos_diff_8, plane_normal))
    !(dot7 == dot8) && return true

    # All edge vertices are on one side of the plane
    return false
end

# 2D
@inline function cell_grid(edge, shape, neighborhood_search::FaceNeighborhoodSearch{2})
    v1, v2 = face_vertices(edge, shape)

    cell1 = cell_coords(v1, neighborhood_search)
    cell2 = cell_coords(v2, neighborhood_search)

    mins = min.(cell1, cell2)
    maxs = max.(cell1, cell2)

    return meshgrid(mins, maxs)
end

# 3D
@inline function cell_grid(triangle, shape, neighborhood_search::FaceNeighborhoodSearch{3})
    v1, v2, v3 = face_vertices(triangle, shape)

    cell1 = cell_coords(v1, neighborhood_search)
    cell2 = cell_coords(v2, neighborhood_search)
    cell3 = cell_coords(v3, neighborhood_search)

    mins = min.(cell1, cell2, cell3)
    maxs = max.(cell1, cell2, cell3)

    return meshgrid(mins, maxs)
end

@inline function meshgrid(min_corner, max_corner; increment=1)
    min_ = collect(min_corner)
    max_ = collect(max_corner)

    ranges = ntuple(dim -> (min_[dim]:increment:max_[dim]), length(min_corner))

    return Iterators.product(ranges...)
end

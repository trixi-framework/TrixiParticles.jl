mutable struct FaceNeighborhoodSearch{NDIMS, CL, ELTYPE} <:
               PointNeighbors.AbstractNeighborhoodSearch
    cell_list     :: CL
    search_radius :: ELTYPE
    periodic_box  :: Nothing # Required by internals of PointNeighbors.jl
    n_cells       :: NTuple{NDIMS, Int}
    cell_size     :: NTuple{NDIMS, ELTYPE} # Required to calculate cell index
end

function FaceNeighborhoodSearch{NDIMS}(; search_radius,
                                       cell_list=PointNeighbors.DictionaryCellList{NDIMS}()) where {NDIMS}
    cell_size = ntuple(_ -> search_radius, Val(NDIMS))
    n_cells = ntuple(_ -> -1, Val(NDIMS))

    return FaceNeighborhoodSearch(cell_list, search_radius, nothing, n_cells, cell_size)
end

@inline Base.ndims(::FaceNeighborhoodSearch{NDIMS}) where {NDIMS} = NDIMS

function faces_in_cell(cell, neighborhood_search)
    return PointNeighbors.points_in_cell(cell, neighborhood_search)
end

@inline function eachneighbor(coords, neighborhood_search::FaceNeighborhoodSearch)
    cell = PointNeighbors.cell_coords(coords, neighborhood_search)
    return faces_in_cell(cell, neighborhood_search)
end

function initialize!(neighborhood_search::FaceNeighborhoodSearch, geometry;
                     pad=ntuple(_ -> 1, ndims(geometry)))
    (; cell_list, search_radius) = neighborhood_search

    empty!(cell_list)

    # Fill cells with intersecting faces
    for face in eachface(geometry)

        # Check if any face intersects a cell in the face-embedding cell grid
        for cell in bounding_box(face, geometry, neighborhood_search)
            if face_intersects_cell(face, geometry, Tuple(cell), neighborhood_search)
                PointNeighbors.push_cell!(cell_list, Tuple(cell), face)
            end
        end
    end

    neighbor_iterator = PointNeighbors.copy_cell_list(cell_list, search_radius, nothing)
    empty!(neighbor_iterator)

    min_cell = PointNeighbors.cell_coords(geometry.min_corner, neighborhood_search) .- pad
    max_cell = PointNeighbors.cell_coords(geometry.max_corner, neighborhood_search) .+ pad

    # Merge all lists of faces in the neighboring cells into one iterator
    face_ids = Int[]
    for cell_runner in meshgrid(min_cell, max_cell)
        resize!(face_ids, 0)
        for neighbor in PointNeighbors.neighboring_cells(cell_runner, neighborhood_search)
            append!(face_ids, faces_in_cell(Tuple(neighbor), neighborhood_search))
        end

        if isempty(face_ids)
            continue
        end

        unique!(face_ids)

        for i in face_ids
            PointNeighbors.push_cell!(neighbor_iterator, Tuple(cell_runner), i)
        end
    end

    neighborhood_search.cell_list = deepcopy(neighbor_iterator)

    return neighborhood_search
end

function face_intersects_cell(face, geometry, cell,
                              neighborhood_search::FaceNeighborhoodSearch{NDIMS}) where {NDIMS}
    (; cell_size) = neighborhood_search

    vertices_list = face_vertices(face, geometry)

    # Check if one of the vertices is inside cell
    for v in vertices_list
        cell == PointNeighbors.cell_coords(v, neighborhood_search) && return true
    end

    # Check if line segments intersect cell
    min_corner = SVector(cell .* cell_size)
    max_corner = min_corner .+ cell_size

    ray_direction = vertices_list[2] - vertices_list[1]
    ray_origin = vertices_list[1]

    ray_intersects_cell(min_corner, max_corner, ray_origin, ray_direction) && return true

    if NDIMS == 3
        ray_direction = vertices_list[2] - vertices_list[3]
        ray_origin = vertices_list[3]

        ray_intersects_cell(min_corner, max_corner, ray_origin, ray_direction) &&
            return true

        ray_direction = vertices_list[3] - vertices_list[1]
        ray_origin = vertices_list[1]

        ray_intersects_cell(min_corner, max_corner, ray_origin, ray_direction) &&
            return true

        # For 3D,  Check if triangle plane intersects cell (for very large triangles)
        normal = face_normal(face, geometry)

        return plane_intersects_cell(ray_origin, normal, min_corner, cell_size)
    end

    return false
end

# See https://tavianator.com/2022/ray_box_boundary.html
function ray_intersects_cell(min_corner, max_corner, ray_origin, ray_direction;
                             pad=sqrt(eps()))
    NDIMS = length(ray_origin)

    inv_dir = SVector(ntuple(@inline(dim->1 / ray_direction[dim]), NDIMS))

    tmin = zero(eltype(ray_direction))
    tmax = Inf
    for dim in 1:NDIMS
        # `pad` is to handle rays on the boundary
        t1 = @inbounds (min_corner[dim] - pad - ray_origin[dim]) * inv_dir[dim]
        t2 = @inbounds (max_corner[dim] + pad - ray_origin[dim]) * inv_dir[dim]

        tmin = min(max(t1, tmin), max(t2, tmin))
        tmax = max(min(t1, tmax), min(t2, tmax))
    end

    return tmin <= tmax
end

# Check if each cell vertex is located on the same side of the plane.
# Otherwise the plane intersects the cell.
function plane_intersects_cell(point_on_plane, plane_normal, cell_min_corner, cell_size)
    cell_center = cell_min_corner .+ cell_size ./ 2

    # `corner1` is the corner that is furthest in the direction of `plane_normal`.
    # When the plane is not intersecting the cell, this corner is the one closest
    # to the plane when the normal is pointing away from the cell or furthest from the plane
    # when the normal is pointing towards the cell.
    # Note that this could also be a face or edge midpoint when the plane is axis-aligned.
    normal_unit = sign.(plane_normal)
    corner1 = cell_center + normal_unit .* cell_size / 2
    corner2 = cell_center - normal_unit .* cell_size / 2

    # These two vectors are on the same side of the plane
    # if and only if the plane intersects the cell
    plane_to_corner1 = corner1 - point_on_plane
    plane_to_corner2 = corner2 - point_on_plane

    # Return true if the two vectors are on different sides of the plane
    return dot(plane_normal, plane_to_corner1) * dot(plane_normal, plane_to_corner2) < 0
end

@inline function bounding_box(face, geometry,
                              neighborhood_search::FaceNeighborhoodSearch{NDIMS}) where {NDIMS}
    vertices = face_vertices(face, geometry)

    # Compute the cell coordinates for each vertex
    cells = (PointNeighbors.cell_coords(v, neighborhood_search) for v in vertices)

    # Compute the element-wise minimum and maximum cell coordinates across all vertices
    mins = reduce((a, b) -> min.(a, b), cells)
    maxs = reduce((a, b) -> max.(a, b), cells)

    return meshgrid(mins, maxs)
end

@inline function meshgrid(min_corner::NTuple{NDIMS}, max_corner) where {NDIMS}
    # In 2D, this returns Cartesian indices
    # {min_corner[1], ..., max_corner[1]} × {min_corner[2], ..., max_corner[2]}.
    return CartesianIndices(ntuple(i -> (min_corner[i]):(max_corner[i]), NDIMS))
end

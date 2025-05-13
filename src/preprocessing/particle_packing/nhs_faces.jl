struct FaceNeighborhoodSearch{NDIMS, CL, ELTYPE} <:
       PointNeighbors.AbstractNeighborhoodSearch
    cell_list     :: CL
    neighbors     :: CL
    search_radius :: ELTYPE
    periodic_box  :: Nothing # Required by internals of PointNeighbors.jl
    n_cells       :: NTuple{NDIMS, Int}
    cell_size     :: NTuple{NDIMS, ELTYPE} # Required to calculate cell index
end

function FaceNeighborhoodSearch{NDIMS}(; search_radius, cell_list) where {NDIMS}
    cell_size = ntuple(_ -> search_radius, Val(NDIMS))
    n_cells = ntuple(_ -> -1, Val(NDIMS))
    cell_list_ = PointNeighbors.copy_cell_list(cell_list, search_radius, nothing)
    neighbors = PointNeighbors.copy_cell_list(cell_list, search_radius, nothing)

    FaceNeighborhoodSearch{NDIMS, typeof(cell_list_),
                           eltype(search_radius)}(cell_list_, neighbors, search_radius,
                                                  nothing, n_cells, cell_size)
end

@inline Base.ndims(::FaceNeighborhoodSearch{NDIMS}) where {NDIMS} = NDIMS

function faces_in_cell(cell, neighborhood_search)
    return PointNeighbors.points_in_cell(cell, neighborhood_search)
end

@inline function eachneighbor(coords, nhs::TrivialNeighborhoodSearch)
    return PointNeighbors.eachneighbor(coords, nhs)
end

@inline function eachneighbor(coords, neighborhood_search::FaceNeighborhoodSearch)
    cell = PointNeighbors.cell_coords(coords, neighborhood_search)
    return neighborhood_search.neighbors[cell]
end

function initialize!(neighborhood_search::FaceNeighborhoodSearch, geometry;
                     pad=ntuple(_ -> 1, ndims(geometry)))
    (; cell_list, neighbors) = neighborhood_search

    empty!(cell_list)

    # Fill cells with intersecting faces
    for face in eachface(geometry)

        # Add face to cells in the face-embedding cell grid
        for cell in bounding_box(face, geometry, neighborhood_search)
            PointNeighbors.push_cell!(cell_list, Tuple(cell), face)
        end
    end

    empty!(neighbors)
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
            PointNeighbors.push_cell!(neighbors, Tuple(cell_runner), i)
        end
    end

    empty!(cell_list)

    return neighborhood_search
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

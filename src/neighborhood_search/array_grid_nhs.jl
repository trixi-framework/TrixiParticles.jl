struct ArrayGridNeighborhoodSearch{NDIMS, ELTYPE, G, GI, HC, PB}
    grid                :: G # Array{Int, NDIMS+1}, [:, i, j, k]
    grid_indices        :: GI # Array{Int, NDIMS} [i, j, k]
    min_corner          :: SVector{NDIMS, ELTYPE}
    max_corner          :: SVector{NDIMS, ELTYPE}
    search_radius       :: ELTYPE
    has_changed         :: HC # Matrix{Bool}, [i, j]
    # cell_buffer         :: Array{NTuple{NDIMS, Int}, 2} # Multithreaded buffer for `update!`
    # cell_buffer_indices :: Vector{Int} # Store which entries of `cell_buffer` are initialized
    periodic_box        :: PB
    n_cells             :: NTuple{NDIMS, Int32}
    cell_size           :: NTuple{NDIMS, ELTYPE}
    threaded_nhs_update :: Bool
end

function ArrayGridNeighborhoodSearch{NDIMS}(search_radius, n_particles;
                                                periodic_box_min_corner=nothing,
                                                periodic_box_max_corner=nothing,
                                                threaded_nhs_update=true) where {NDIMS}
        ELTYPE = typeof(search_radius)

        min_corner = SVector(-4.0, -4.0)
        max_corner = SVector(8.0, 9.0)
        buffer_size = 40

        # Round up search radius so that the grid fits exactly into the domain without
        # splitting any cells. This might impact performance slightly, since larger
        # cells mean that more potential neighbors are considered than necessary.
        # Allow small tolerance to avoid inefficient larger cells due to machine
        # rounding errors.
        size = max_corner - min_corner
        n_cells = Tuple(floor.(Int32, (size .+ 10eps()) / search_radius))
        cell_size = Tuple(size ./ n_cells)

        grid = Array{Int32, NDIMS + 1}(undef, buffer_size, n_cells...)
        grid_indices = zeros(Int32, n_cells...)

        # cell_buffer = Array{NTuple{NDIMS, Int}, 2}(undef, n_particles, Threads.nthreads())
        # cell_buffer_indices = zeros(Int, Threads.nthreads())
        has_changed = Matrix{Bool}(undef, n_cells...)

        if search_radius < eps() ||
           (periodic_box_min_corner === nothing && periodic_box_max_corner === nothing)

            # No periodicity
            periodic_box = nothing
        elseif periodic_box_min_corner !== nothing && periodic_box_max_corner !== nothing
            periodic_box = PeriodicBox(periodic_box_min_corner, periodic_box_max_corner)

            if any(i -> i < 3, n_cells)
                throw(ArgumentError("the `ArrayGridNeighborhoodSearch` needs at least 3 cells " *
                                    "in each dimension when used with periodicity. " *
                                    "Please use no NHS for very small problems."))
            end
        else
            throw(ArgumentError("`periodic_box_min_corner` and `periodic_box_max_corner` " *
                                "must either be both `nothing` or both an array or tuple"))
        end

        ArrayGridNeighborhoodSearch{NDIMS, ELTYPE, typeof(grid), typeof(grid_indices),
                                    typeof(has_changed),
            typeof(periodic_box)}(grid, grid_indices, min_corner, max_corner,
                                  search_radius, has_changed, #cell_buffer, cell_buffer_indices,
                                  periodic_box, n_cells, cell_size, threaded_nhs_update)
    end

@inline Base.ndims(neighborhood_search::ArrayGridNeighborhoodSearch{NDIMS}) where {NDIMS} = NDIMS

function initialize!(neighborhood_search::ArrayGridNeighborhoodSearch, ::Nothing)
    # No particle coordinates function -> don't initialize.
    return neighborhood_search
end

function initialize!(neighborhood_search::ArrayGridNeighborhoodSearch, coords_fun)
    error("use array instead")
end

function initialize!(neighborhood_search::ArrayGridNeighborhoodSearch, coords::AbstractArray)
    (; grid, grid_indices) = neighborhood_search

    grid .= 0
    grid_indices .= 0

    CUDA.@allowscalar for particle in 1:size(coords, 2)
        # Get cell index of the particle's cell
        cell = cell_coords(extract_svector(coords, neighborhood_search, particle),
                           neighborhood_search)

        # Add particle to corresponding cell
        grid_indices[cell...] += 1
        grid[grid_indices[cell...], cell...] = particle
    end

    return neighborhood_search
end

function update!(neighborhood_search::ArrayGridNeighborhoodSearch, ::Nothing)
    # No particle coordinates function -> don't update.
    return neighborhood_search
end

function update!(neighborhood_search::ArrayGridNeighborhoodSearch, coords_fun)
    # initialize!(neighborhood_search, coords)
    error("use array instead")
end

# Modify the existing hash table by moving particles into their new cells
function update!(neighborhood_search::ArrayGridNeighborhoodSearch, coords::AbstractArray)
    # initialize!(neighborhood_search, coords)
    (; grid, grid_indices, has_changed) = neighborhood_search

    # Find all cells containing particles that now belong to another cell
    # @trixi_timeit timer() "mark changed cells" begin
    #     has_changed .= false
    #     backend = get_backend(grid)
    #     mark_changed_cells_kernel!(backend)(neighborhood_search, coords, ndrange=size(grid_indices))
    #     synchronize(backend)
    # end

    # @trixi_timeit timer() "move to new cells" begin
    #     # Move right
    #     backend = get_backend(grid)
    #     move_cell_kernel!(backend)(neighborhood_search, coords, (1, 0), ndrange=size(grid_indices))
    #     synchronize(backend)

    #     # Move left
    #     backend = get_backend(grid)
    #     move_cell_kernel!(backend)(neighborhood_search, coords, (-1, 0), ndrange=size(grid_indices))
    #     synchronize(backend)

    #     # Move up
    #     backend = get_backend(grid)
    #     move_cell_kernel!(backend)(neighborhood_search, coords, (0, 1), ndrange=size(grid_indices))
    #     synchronize(backend)

    #     # Move down
    #     backend = get_backend(grid)
    #     move_cell_kernel!(backend)(neighborhood_search, coords, (0, -1), ndrange=size(grid_indices))
    #     synchronize(backend)

    #     # Move diagonally
    #     backend = get_backend(grid)
    #     move_cell_kernel!(backend)(neighborhood_search, coords, (1, 1), ndrange=size(grid_indices))
    #     synchronize(backend)
    #     backend = get_backend(grid)
    #     move_cell_kernel!(backend)(neighborhood_search, coords, (-1, 1), ndrange=size(grid_indices))
    #     synchronize(backend)
    #     backend = get_backend(grid)
    #     move_cell_kernel!(backend)(neighborhood_search, coords, (1, -1), ndrange=size(grid_indices))
    #     synchronize(backend)
    #     backend = get_backend(grid)
    #     move_cell_kernel!(backend)(neighborhood_search, coords, (-1, -1), ndrange=size(grid_indices))
    #     synchronize(backend)
    # end

    # Iterate over all marked cells and move the particles into their new cells
    # @trixi_timeit timer() "move to new cells" begin
    #     CUDA.@allowscalar while findmax(has_changed)[1]
    #         cell = findmax(has_changed)[2]
    #         has_changed[cell] = false

    #         # Find all particles whose coordinates do not match this cell
    #         moved_particle_indices = (i for i in 1:grid_indices[cell]
    #                                 if cell_coords(extract_svector(coords, neighborhood_search, grid[i, cell]),
    #                                                 neighborhood_search) != Tuple(cell))

    #         # Add moved particles to new cell
    #         for i in moved_particle_indices
    #             particle = grid[i, cell]
    #             new_cell_coords = cell_coords(extract_svector(coords, neighborhood_search, particle), neighborhood_search)

    #             # Add particle to new cell
    #             grid_indices[new_cell_coords...] += 1
    #             grid[grid_indices[new_cell_coords...], new_cell_coords...] = particle
    #         end
    #     end
    # end

    @trixi_timeit timer() "move atomic" begin
        backend = get_backend(grid)
        move_cell_kernel2!(backend)(neighborhood_search, coords, ndrange=size(grid_indices))
        synchronize(backend)
    end

    @trixi_timeit timer() "remove from old cells" begin
        backend = get_backend(grid)
        remove_from_cells_kernel!(backend)(neighborhood_search, coords, ndrange=size(grid_indices))
        synchronize(backend)
    end

    # @trixi_timeit timer() "sanity check" begin
    #     has_changed .= 0
    #     backend = get_backend(grid)
    #     # Check that all particles are in the correct cell
    #     sanity_check_kernel1!(backend)(neighborhood_search, coords, ndrange=size(coords, 2))
    #     # Check that each cell doesn't contain extra particles that don't belong there
    #     sanity_check_kernel2!(backend)(neighborhood_search, coords, ndrange=size(coords, 2))
    #     synchronize(backend)

    #     if findmax(has_changed)[1]
    #         @warn "NHS re-initialization"
    #         @trixi_timeit timer() "nhs re-initialization" initialize!(neighborhood_search, coords)
    #     end
    # end

    return neighborhood_search
end

@kernel function mark_changed_cells_kernel!(neighborhood_search, coords)
    (; has_changed) = neighborhood_search

    cell = @index(Global, NTuple)

    for particle in particles_in_cell(cell, neighborhood_search)
        new_cell = cell_coords(extract_svector(coords, neighborhood_search, particle), neighborhood_search)
        if cell != new_cell
            # Mark this cell and continue with the next one
            has_changed[cell...] = true
            break
        end
    end
end

@kernel function move_cell_kernel2!(neighborhood_search, coords)
    (; grid, grid_indices, has_changed) = neighborhood_search

    cell = @index(Global, NTuple)

    for particle in particles_in_cell(cell, neighborhood_search)
        new_cell = cell_coords(extract_svector(coords, neighborhood_search, particle), neighborhood_search)
        if cell != new_cell
            # Add moved particle to new cell
            index = CUDA.@atomic grid_indices[new_cell...] += 1

            # TODO why is index the old `grid_indices[new_cell...]`
            # and not the incremented value?
            grid[index + 1, new_cell...] = particle
        end
    end
end

@kernel function move_cell_kernel!(neighborhood_search, coords, move_step)
    (; grid, grid_indices, has_changed) = neighborhood_search

    cell = @index(Global, NTuple)

    if has_changed[cell...]
        for particle in particles_in_cell(cell, neighborhood_search)
            new_cell = cell_coords(extract_svector(coords, neighborhood_search, particle), neighborhood_search)
            if cell .+ move_step == new_cell
                # Add moved particle to new cell
                grid_indices[new_cell...] += 1
                grid[grid_indices[new_cell...], new_cell...] = particle
            end
        end
    end
end

@kernel function remove_from_cells_kernel!(neighborhood_search, coords)
    (; grid, grid_indices) = neighborhood_search

    cell = @index(Global, NTuple)

    # Remove moved particles from old cell
    for i in grid_indices[cell...]:-1:1
        # Remove moved particles from the end of the cell
        new_cell = cell_coords(extract_svector(coords, neighborhood_search, grid[i, cell...]),
                               neighborhood_search)
        if new_cell != cell
            grid[i, cell...] = grid[grid_indices[cell...], cell...]
            grid[grid_indices[cell...], cell...] = 0
            grid_indices[cell...] -= 1
        end
    end
end

@kernel function sanity_check_kernel1!(neighborhood_search, coords)
    (; grid, grid_indices) = neighborhood_search

    particle = @index(Global)

    cell = cell_coords(extract_svector(coords, neighborhood_search, particle),
                       neighborhood_search)
    if !(particle in view(grid, 1:grid_indices[cell...], cell...))
        neighborhood_search.has_changed[cell...] = 1
    end
end

@kernel function sanity_check_kernel2!(neighborhood_search, coords)
    (; grid, grid_indices) = neighborhood_search

    cell = @index(Global, NTuple)

    for i in 1:grid_indices[cell...]
        # Remove moved particles from the end of the cell
        new_cell = cell_coords(extract_svector(coords, neighborhood_search, grid[i, cell...]),
                               neighborhood_search)
        if new_cell != cell
            neighborhood_search.has_changed[cell...] = 1
        end
    end
end

# 1D
@inline function eachneighbor(coords, neighborhood_search::ArrayGridNeighborhoodSearch{1})
    cell = cell_coords(coords, neighborhood_search)
    x = cell[1]
    # Generator of all neighboring cells to consider
    neighboring_cells = ((x + i) for i in -1:1)

    # Merge all lists of particles in the neighboring cells into one iterator
    Iterators.flatten(particles_in_cell(cell, neighborhood_search)
                      for cell in neighboring_cells)
end

# 2D
@inline function eachneighbor(coords, neighborhood_search::ArrayGridNeighborhoodSearch{2})
    cell = cell_coords(coords, neighborhood_search)
    x, y = cell
    # Generator of all neighboring cells to consider
    neighboring_cells = ((x + i, y + j) for i in -1:1, j in -1:1)

    # Merge all lists of particles in the neighboring cells into one iterator
    Iterators.flatten(particles_in_cell(cell, neighborhood_search)
                      for cell in neighboring_cells)
end

# 3D
@inline function eachneighbor(coords, neighborhood_search::ArrayGridNeighborhoodSearch{3})
    cell = cell_coords(coords, neighborhood_search)
    x, y, z = cell
    # Generator of all neighboring cells to consider
    neighboring_cells = ((x + i, y + j, z + k) for i in -1:1, j in -1:1, k in -1:1)

    # Merge all lists of particles in the neighboring cells into one iterator
    Iterators.flatten(particles_in_cell(cell, neighborhood_search)
                      for cell in neighboring_cells)
end

@inline function particles_in_cell(cell_index, neighborhood_search::ArrayGridNeighborhoodSearch)
    (; grid, grid_indices) = neighborhood_search

    cell = periodic_cell_index(cell_index, neighborhood_search)

    return view(grid, 1:grid_indices[cell...], cell...)
end

# @inline function periodic_cell_index(cell_index, neighborhood_search)
#     (; n_cells, periodic_box) = neighborhood_search

#     periodic_cell_index(cell_index, periodic_box, n_cells)
# end

# @inline periodic_cell_index(cell_index, ::Nothing, n_cells) = cell_index

# @inline function periodic_cell_index(cell_index, periodic_box, n_cells)
#     return rem.(cell_index, n_cells, RoundDown)
# end

@inline function cell_coords(coords, neighborhood_search::ArrayGridNeighborhoodSearch)
    (; periodic_box, min_corner, cell_size) = neighborhood_search

    coords_ = periodic_coords(coords, periodic_box) .- min_corner

    return cell_coords(coords_, nothing, cell_size) .+ 1
end

# @inline function cell_coords(coords, periodic_box::Nothing, cell_size)
#     return Tuple(floor_to_int.(coords ./ cell_size))
# end

# @inline function cell_coords(coords, periodic_box, cell_size)
#     # Subtract `min_corner` to offset coordinates so that the min corner of the periodic
#     # box corresponds to the (0, 0) cell of the NHS.
#     # This way, there are no partial cells in the domain if the domain size is an integer
#     # multiple of the cell size (which is required, see the constructor).
#     offset_coords = periodic_coords(coords, periodic_box) .- periodic_box.min_corner

#     return Tuple(floor_to_int.(offset_coords ./ cell_size))
# end

# # When particles end up with coordinates so big that the cell coordinates
# # exceed the range of Int, then `floor(Int, i)` will fail with an InexactError.
# # In this case, we can just use typemax(Int), since we can assume that particles
# # that far away will not interact with anything, anyway.
# # This usually indicates an instability, but we don't want the simulation to crash,
# # since adaptive time integration methods may detect the instability and reject the
# # time step.
# # If we threw an error here, we would prevent the time integration method from
# # retrying with a smaller time step, and we would thus crash perfectly fine simulations.
# @inline function floor_to_int(i)
#     if isnan(i) || i > typemax(Int)
#         return typemax(Int)
#     elseif i < typemin(Int)
#         return typemin(Int)
#     end

#     return floor(Int, i)
# end

# # Sorting only really makes sense in longer simulations where particles
# # end up very unordered.
# # WARNING: This is currently unmaintained.
# function z_index_sort!(coordinates, system)
#     (; mass, pressure, neighborhood_search) = system

#     perm = sortperm(eachparticle(system),
#                     by=(i -> cell_z_index(extract_svector(coordinates, system, i),
#                                           neighborhood_search)))

#     permute!(mass, perm)
#     permute!(pressure, perm)
#     Base.permutecols!!(u, perm)

#     return nothing
# end

# @inline function cell_z_index(coords, neighborhood_search)
#     cell = cell_coords(coords, neighborhood_search.search_radius) .+ 1

#     return cartesian2morton(SVector(cell))
# end

# # Create a copy of a neighborhood search but with a different search radius
# function copy_neighborhood_search(nhs::ArrayGridNeighborhoodSearch, search_radius, u)
#     if nhs.periodic_box === nothing
#         search = ArrayGridNeighborhoodSearch{ndims(nhs)}(search_radius, nparticles(nhs))
#     else
#         search = ArrayGridNeighborhoodSearch{ndims(nhs)}(search_radius, nparticles(nhs),
#                                                     periodic_box_min_corner=nhs.periodic_box.min_corner,
#                                                     periodic_box_max_corner=nhs.periodic_box.max_corner)
#     end

#     # Initialize neighborhood search
#     initialize!(search, u)

#     return search
# end

# # Create a copy of a neighborhood search but with a different search radius
# function copy_neighborhood_search(nhs::TrivialNeighborhoodSearch, search_radius, u)
#     return nhs
# end

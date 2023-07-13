struct TrivialNeighborhoodSearch{NDIMS, ELTYPE, EP}
    search_radius::ELTYPE
    eachparticle::EP
    periodic_box_size::Nothing

    function TrivialNeighborhoodSearch{NDIMS}(search_radius, eachparticle) where {NDIMS}
        new{NDIMS, typeof(search_radius), typeof(eachparticle)}(search_radius, eachparticle,
                                                                nothing)
    end
end

@inline function Base.ndims(neighborhood_search::TrivialNeighborhoodSearch{NDIMS}) where {
                                                                                          NDIMS
                                                                                          }
    return NDIMS
end

@inline initialize!(search::TrivialNeighborhoodSearch, coords_fun) = search
@inline update!(search::TrivialNeighborhoodSearch, coords_fun) = search
@inline eachneighbor(coords, search::TrivialNeighborhoodSearch) = search.eachparticle

@doc raw"""
    SpatialHashingSearch{NDIMS}(search_radius, n_particles)

Simple grid-based neighborhood search with uniform search radius ``d``,
inspired by (Ihmsen et al. 2011, Section 4.4).

The domain is divided into cells of uniform size ``d`` in each dimension.
Only particles in neighboring cells are then considered as neighbors of a particle.
Instead of representing a finite domain by an array of cells (basic uniform grid),
a potentially infinite domain is represented by saving cells in a hash table,
indexed by the cell index tuple
```math
\left( \left\lfloor \frac{x}{d} \right\rfloor, \left\lfloor \frac{y}{d} \right\rfloor \right) \quad \text{or} \quad
\left( \left\lfloor \frac{x}{d} \right\rfloor, \left\lfloor \frac{y}{d} \right\rfloor, \left\lfloor \frac{z}{d} \right\rfloor \right),
```
where ``x, y, z`` are the space coordinates.

As opposed to (Ihmsen et al. 2011), we do not handle the hashing explicitly and use
Julia's `Dict` data structure instead.
We also do not sort the particles in any way, since that makes our implementation
a lot faster (although not parallelizable).

## References:
- Markus Ihmsen, Nadir Akinci, Markus Becker, Matthias Teschner.
  "A Parallel SPH Implementation on Multi-Core CPUs".
  In: Computer Graphics Forum 30.1 (2011), pages 99–112.
  [doi: 10.1111/J.1467-8659.2010.01832.X](https://doi.org/10.1111/J.1467-8659.2010.01832.X)
"""
struct SpatialHashingSearch{NDIMS, ELTYPE, PBS}
    hashtable             :: Dict{NTuple{NDIMS, Int}, Vector{Int}}
    search_radius         :: ELTYPE
    empty_vector          :: Vector{Int} # Just an empty vector (used in `eachneighbor`)
    cell_buffer           :: Array{NTuple{NDIMS, Int}, 2} # Multithreaded buffer for `update!`
    cell_buffer_indices   :: Vector{Int} # Store which entries of `cell_buffer` are initialized
    periodic_box_size     :: PBS # `SVector{NDIMS, ELTYPE}` or `Nothing` without periodicity
    bound_and_ghost_cells :: Vector{NTuple{NDIMS, Int}}

    function SpatialHashingSearch{NDIMS}(search_radius, n_particles;
                                         min_corner=nothing,
                                         max_corner=nothing) where {NDIMS}
        ELTYPE = typeof(search_radius)

        hashtable = Dict{NTuple{NDIMS, Int}, Vector{Int}}()
        empty_vector = Int[]
        cell_buffer = Array{NTuple{NDIMS, Int}, 2}(undef, n_particles, Threads.nthreads())
        cell_buffer_indices = zeros(Int, Threads.nthreads())

        if (min_corner === nothing && max_corner === nothing) || search_radius < eps()
            # No periodicity
            periodic_box_size = nothing
            bound_and_ghost_cells = NTuple{NDIMS, Int}[]
        elseif min_corner !== nothing && max_corner !== nothing
            if NDIMS == 3
                throw(ArgumentError("periodic neighborhood search is not yet supported in 3D"))
            end

            periodic_box_size = SVector(Tuple(max_corner - min_corner))

            # If box size is not an integer multiple of search radius
            if !all(abs.(rem.(periodic_box_size / search_radius, 1, RoundNearest)) .< 1e-5)
                # TODO allow other domain sizes
                throw(ArgumentError("size of the periodic box must be an integer multiple " *
                                    "of `search_radius`"))
            end

            min_cell = cell_coords(min_corner .+
                                   0.5 * search_radius * ones(SVector{NDIMS, ELTYPE}),
                                   search_radius)
            max_cell = cell_coords(max_corner .-
                                   0.5 * search_radius * ones(SVector{NDIMS, ELTYPE}),
                                   search_radius)
            bound_and_ghost_cells = initialize_boundary_and_ghost_cells(hashtable,
                                                                        min_cell, max_cell)
        else
            throw(ArgumentError("`min_corner` and `max_corner` must either be " *
                                "both `nothing` or both an array or tuple"))
        end

        new{NDIMS, ELTYPE,
            typeof(periodic_box_size)}(hashtable, search_radius, empty_vector,
                                       cell_buffer, cell_buffer_indices,
                                       periodic_box_size, bound_and_ghost_cells)
    end
end

function initialize_boundary_and_ghost_cells(hashtable, min_cell, max_cell)
    # To make sure that the hashtable starts empty.
    # Then, we can just return the keyset after we initialized all boundary and ghost cells.
    empty!(hashtable)

    # Initialize boundary and ghost cells
    for y in min_cell[2]:max_cell[2]
        # -x boundary cells
        hashtable[(min_cell[1], y)] = Int[]

        # +x boundary cells
        hashtable[(max_cell[1], y)] = Int[]

        # -x ghost cells
        hashtable[(min_cell[1] - 1, y)] = hashtable[(max_cell[1], y)]

        # +x ghost cells
        hashtable[(max_cell[1] + 1, y)] = hashtable[(min_cell[1], y)]
    end

    for x in min_cell[1]:max_cell[1]
        # Avoid setting corner boundary cells again
        if min_cell[1] < x < max_cell[1]
            # -y boundary cells
            hashtable[(x, min_cell[2])] = Int[]

            # +y boundary cells
            hashtable[(x, max_cell[2])] = Int[]
        end

        # -y ghost cells
        hashtable[(x, min_cell[2] - 1)] = hashtable[(x, max_cell[2])]

        # +y ghost cells
        hashtable[(x, max_cell[2] + 1)] = hashtable[(x, min_cell[2])]
    end

    # Corner ghost cells
    hashtable[(min_cell[1] - 1, min_cell[2] - 1)] = hashtable[(max_cell[1], max_cell[2])]
    hashtable[(max_cell[1] + 1, min_cell[2] - 1)] = hashtable[(min_cell[1], max_cell[2])]
    hashtable[(min_cell[1] - 1, max_cell[2] + 1)] = hashtable[(max_cell[1], min_cell[2])]
    hashtable[(max_cell[1] + 1, max_cell[2] + 1)] = hashtable[(min_cell[1], min_cell[2])]

    return collect(keys(hashtable))
end

@inline Base.ndims(neighborhood_search::SpatialHashingSearch{NDIMS}) where {NDIMS} = NDIMS

@inline function nparticles(neighborhood_search::SpatialHashingSearch)
    return size(neighborhood_search.cell_buffer, 1)
end

function initialize!(neighborhood_search::SpatialHashingSearch, ::Nothing)
    # No particle coordinates function -> don't initialize.
    return neighborhood_search
end

function initialize!(neighborhood_search::SpatialHashingSearch{NDIMS},
                     x::AbstractArray) where {NDIMS}
    initialize!(neighborhood_search, i -> extract_svector(x, Val(NDIMS), i))
end

function initialize!(neighborhood_search::SpatialHashingSearch, coords_fun)
    @unpack hashtable, search_radius, bound_and_ghost_cells = neighborhood_search

    # Delete all cells that are not boundary or ghost cells
    for cell in keys(hashtable)
        if !(cell in bound_and_ghost_cells)
            delete!(hashtable, cell)
        end
    end

    # Empty boundary cells
    for cell in bound_and_ghost_cells
        empty!(hashtable[cell])
    end

    # This is needed to prevent lagging on macOS ARM.
    # See https://github.com/JuliaSIMD/Polyester.jl/issues/89
    ThreadingUtilities.sleep_all_tasks()

    for particle in 1:nparticles(neighborhood_search)
        # Get cell index of the particle's cell
        cell = cell_coords(coords_fun(particle), search_radius)

        # Add particle to corresponding cell or create cell if it does not exist
        if haskey(hashtable, cell)
            append!(hashtable[cell], particle)
        else
            hashtable[cell] = [particle]
        end
    end

    return neighborhood_search
end

function update!(neighborhood_search::SpatialHashingSearch, ::Nothing)
    # No particle coordinates function -> don't update.
    return neighborhood_search
end

function update!(neighborhood_search::SpatialHashingSearch{NDIMS},
                 x::AbstractArray) where {NDIMS}
    update!(neighborhood_search, i -> extract_svector(x, Val(NDIMS), i))
end

# Modify the existing hash table by moving particles into their new cells
function update!(neighborhood_search::SpatialHashingSearch, coords_fun)
    @unpack hashtable, search_radius, cell_buffer, cell_buffer_indices,
    bound_and_ghost_cells = neighborhood_search

    # Reset `cell_buffer` by moving all pointers to the beginning.
    cell_buffer_indices .= 0

    # Find all cells containing particles that now belong to another cell.
    # `collect` the keyset to be able to loop over it with `@threaded`.
    @threaded for cell in collect(keys(hashtable))
        mark_changed_cell!(neighborhood_search, cell, coords_fun)
    end

    # This is needed to prevent lagging on macOS ARM.
    # See https://github.com/JuliaSIMD/Polyester.jl/issues/89
    ThreadingUtilities.sleep_all_tasks()

    # Iterate over all marked cells and move the particles into their new cells.
    for thread in 1:Threads.nthreads()
        # Only the entries `1:cell_buffer_indices[thread]` are initialized for `thread`.
        for i in 1:cell_buffer_indices[thread]
            cell = cell_buffer[i, thread]
            particles = hashtable[cell]

            # Find all particles whose coordinates do not match this cell
            moved_particle_indices = (i for i in eachindex(particles)
                                      if cell_coords(coords_fun(particles[i]),
                                                     search_radius) != cell)

            # Add moved particles to new cell
            for i in moved_particle_indices
                particle = particles[i]
                new_cell_coords = cell_coords(coords_fun(particle), search_radius)

                # Add particle to corresponding cell or create cell if it does not exist
                if haskey(hashtable, new_cell_coords)
                    append!(hashtable[new_cell_coords], particle)
                else
                    hashtable[new_cell_coords] = [particle]
                end
            end

            # Remove moved particles from this cell or delete the cell if it is now empty
            if !(cell in bound_and_ghost_cells) &&
               count(_ -> true, moved_particle_indices) == length(particles)
                delete!(hashtable, cell)
            else
                deleteat!(particles, moved_particle_indices)
            end
        end
    end

    return neighborhood_search
end

# Use this function barrier and unpack inside to avoid passing closures to Polyester.jl
# with @batch (@threaded).
# Otherwise, @threaded does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
@inline function mark_changed_cell!(neighborhood_search, cell, coords_fun)
    @unpack hashtable, search_radius, cell_buffer, cell_buffer_indices = neighborhood_search

    for particle in hashtable[cell]
        if cell_coords(coords_fun(particle), search_radius) != cell
            # Mark this cell and continue with the next one.
            #
            # `cell_buffer` is preallocated,
            # but only the entries 1:i are used for this thread.
            i = cell_buffer_indices[Threads.threadid()] += 1
            cell_buffer[i, Threads.threadid()] = cell
            break
        end
    end
end

@inline function eachneighbor(coords, neighborhood_search::SpatialHashingSearch{2})
    cell = cell_coords(coords, neighborhood_search.search_radius)
    x, y = cell
    # Generator of all neighboring cells to consider
    neighboring_cells = ((x + i, y + j) for i in -1:1, j in -1:1)

    # Merge all lists of particles in the neighboring cells into one iterator
    Iterators.flatten(particles_in_cell(cell, neighborhood_search)
                      for cell in neighboring_cells)
end

@inline function eachneighbor(coords, neighborhood_search::SpatialHashingSearch{3})
    cell = cell_coords(coords, neighborhood_search.search_radius)
    x, y, z = cell
    # Generator of all neighboring cells to consider
    neighboring_cells = ((x + i, y + j, z + k) for i in -1:1, j in -1:1, k in -1:1)

    # Merge all lists of particles in the neighboring cells into one iterator
    Iterators.flatten(particles_in_cell(cell, neighborhood_search)
                      for cell in neighboring_cells)
end

# Loop over all pairs of particles and neighbors within the kernel cutoff.
# `f(particle, neighbor, pos_diff, distance)` is called for every particle-neighbor pair.
# By default, loop over `each_moving_particle(system)`.
@inline function for_particle_neighbor(f, system, neighbor_system,
                                       system_coords, neighbor_coords, neighborhood_search;
                                       particles=each_moving_particle(system))
    @threaded for particle in particles
        for_particle_neighbor_inner(f, system_coords, neighbor_coords, neighborhood_search,
                                    particle)
    end

    return nothing
end

# Use this function barrier and unpack inside to avoid passing closures to Polyester.jl
# with @batch (@threaded).
# Otherwise, @threaded does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
@inline function for_particle_neighbor_inner(f, system_coords, neighbor_system_coords,
                                             neighborhood_search, particle)
    @unpack search_radius, periodic_box_size = neighborhood_search

    particle_coords = extract_svector(system_coords, Val(ndims(neighborhood_search)),
                                      particle)
    for neighbor in eachneighbor(particle_coords, neighborhood_search)
        neighbor_coords = extract_svector(neighbor_system_coords,
                                          Val(ndims(neighborhood_search)), neighbor)

        pos_diff = particle_coords - neighbor_coords
        distance2 = dot(pos_diff, pos_diff)

        pos_diff, distance2 = compute_periodic_distance(pos_diff, distance2, search_radius,
                                                        periodic_box_size)

        @autoinfiltrate

        if distance2 <= search_radius^2
            distance = sqrt(distance2)

            # Inline to avoid loss of performance
            # compared to not using `for_particle_neighbor`.
            @inline f(particle, neighbor, pos_diff, distance)
        end
    end
end

@inline function compute_periodic_distance(pos_diff, distance2, search_radius,
                                           periodic_box_size::Nothing)
    return pos_diff, distance2
end

@inline function compute_periodic_distance(pos_diff, distance2, search_radius,
                                           periodic_box_size)
    if distance2 > search_radius^2
        # Use periodic `pos_diff`
        pos_diff -= periodic_box_size .* round.(pos_diff ./ periodic_box_size)
        distance2 = dot(pos_diff, pos_diff)
    end

    return pos_diff, distance2
end

@inline function particles_in_cell(cell_index, neighborhood_search)
    @unpack hashtable, empty_vector = neighborhood_search

    # Return an empty vector when `cell_index` is not a key of `hashtable` and
    # reuse the empty vector to avoid allocations
    return get(hashtable, cell_index, empty_vector)
end

@inline function cell_coords(coords, search_radius)
    return Tuple(floor_to_int.(coords / search_radius))
end

# When particles end up with coordinates so big that the cell coordinates
# exceed the range of Int, then `floor(Int, i)` will fail with an InexactError.
# In this case, we can just use typemax(Int), since we can assume that particles
# that far away will not interact with anything, anyway.
# This usually indicates an instability, but we don't want the simulation to crash,
# since adaptive time integration methods may detect the instability and reject the
# time step.
# If we threw an error here, we would prevent the time integration method from
# retrying with a smaller time step, and we would thus crash perfectly fine simulations.
@inline function floor_to_int(i)
    if isnan(i) || i > typemax(Int)
        return typemax(Int)
    elseif i < typemin(Int)
        return typemin(Int)
    end

    return floor(Int, i)
end

# Sorting only really makes sense in longer simulations where particles
# end up very unordered.
# WARNING: This is currently unmaintained.
function z_index_sort!(coordinates, system)
    @unpack mass, pressure, neighborhood_search = system

    perm = sortperm(eachparticle(system),
                    by=(i -> cell_z_index(extract_svector(coordinates, system, i),
                                          neighborhood_search)))

    permute!(mass, perm)
    permute!(pressure, perm)
    Base.permutecols!!(u, perm)

    return nothing
end

@inline function cell_z_index(coords, neighborhood_search)
    cell = cell_coords(coords, neighborhood_search.search_radius) .+ 1

    return cartesian2morton(SVector(cell))
end

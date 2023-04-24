struct TrivialNeighborhoodSearch{E}
    eachparticle::E

    function TrivialNeighborhoodSearch(container)
        new{typeof(eachparticle(container))}(eachparticle(container))
    end
end

@inline initialize!(search::TrivialNeighborhoodSearch, u, container) = nothing
@inline update!(search::TrivialNeighborhoodSearch, u, container) = nothing
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
struct SpatialHashingSearch{NDIMS, ELTYPE}
    hashtable           :: Dict{NTuple{NDIMS, Int}, Vector{Int}}
    search_radius       :: ELTYPE
    empty_vector        :: Vector{Int} # Just an empty vector (used in `eachneighbor`)
    cell_buffer         :: Array{NTuple{NDIMS, Int}, 2} # Multithreaded buffer for `update!`
    cell_buffer_indices :: Vector{Int} # Store which entries of `cell_buffer` are initialized

    function SpatialHashingSearch{NDIMS}(search_radius, n_particles) where {NDIMS}
        hashtable = Dict{NTuple{NDIMS, Int}, Vector{Int}}()
        empty_vector = Vector{Int}()
        cell_buffer = Array{NTuple{NDIMS, Int}, 2}(undef, n_particles, Threads.nthreads())
        cell_buffer_indices = zeros(Int, Threads.nthreads())

        new{NDIMS, typeof(search_radius)}(hashtable, search_radius, empty_vector,
                                          cell_buffer, cell_buffer_indices)
    end
end

function initialize!(neighborhood_search::SpatialHashingSearch, coordinates, container)
    @unpack hashtable = neighborhood_search

    empty!(hashtable)

    # This is needed to prevent lagging on macOS ARM.
    # See https://github.com/JuliaSIMD/Polyester.jl/issues/89
    ThreadingUtilities.sleep_all_tasks()

    for particle in eachparticle(container)
        # Get cell index of the particle's cell
        cell = cell_coords(extract_svector(coordinates, container, particle),
                           neighborhood_search)

        # Add particle to corresponding cell or create cell if it does not exist
        if haskey(hashtable, cell)
            append!(hashtable[cell], particle)
        else
            hashtable[cell] = [particle]
        end
    end

    return neighborhood_search
end

# Modify the existing hash table by moving particles into their new cells
function update!(neighborhood_search::SpatialHashingSearch, coordinates, container)
    @unpack hashtable, cell_buffer, cell_buffer_indices = neighborhood_search

    @inline function cell_coords(particle)
        cell_coords(extract_svector(coordinates, container, particle),
                    neighborhood_search)
    end

    # Reset `cell_buffer` by moving all pointers to the beginning.
    cell_buffer_indices .= 0

    # Find all cells containing particles that now belong to another cell.
    # `collect` the keyset to be able to loop over it with `@threaded`.
    @threaded for cell in collect(keys(hashtable))
        for particle in hashtable[cell]
            if cell_coords(particle) != cell
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
                                      if cell_coords(particles[i]) != cell)

            # Add moved particles to new cell
            for i in moved_particle_indices
                particle = particles[i]
                new_cell_coords = cell_coords(particle)

                # Add particle to corresponding cell or create cell if it does not exist
                if haskey(hashtable, new_cell_coords)
                    append!(hashtable[new_cell_coords], particle)
                else
                    hashtable[new_cell_coords] = [particle]
                end
            end

            # Remove moved particles from this cell or delete the cell if it is now empty
            if count(_ -> true, moved_particle_indices) == length(particles)
                delete!(hashtable, cell)
            else
                deleteat!(particles, moved_particle_indices)
            end
        end
    end

    return neighborhood_search
end

@inline function eachneighbor(coords, neighborhood_search::SpatialHashingSearch{2})
    cell = cell_coords(coords, neighborhood_search)
    x, y = cell
    # Generator of all neighboring cells to consider
    neighboring_cells = ((x + i, y + j) for i in -1:1, j in -1:1)

    # Merge all lists of particles in the neighboring cells into one iterator
    Iterators.flatten(particles_in_cell(cell, neighborhood_search)
                      for cell in neighboring_cells)
end

@inline function eachneighbor(coords, neighborhood_search::SpatialHashingSearch{3})
    cell = cell_coords(coords, neighborhood_search)
    x, y, z = cell
    # Generator of all neighboring cells to consider
    neighboring_cells = ((x + i, y + j, z + k) for i in -1:1, j in -1:1, k in -1:1)

    # Merge all lists of particles in the neighboring cells into one iterator
    Iterators.flatten(particles_in_cell(cell, neighborhood_search)
                      for cell in neighboring_cells)
end

@inline function particles_in_cell(cell_index, neighborhood_search)
    @unpack hashtable, empty_vector = neighborhood_search

    # Return an empty vector when `cell_index` is not a key of `hastable` and
    # reuse the empty vector to avoid allocations
    return get(hashtable, cell_index, empty_vector)
end

@inline function cell_coords(coords, neighborhood_search)
    @unpack search_radius = neighborhood_search

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
# end up very unordered
function z_index_sort!(coordinates, container)
    @unpack mass, pressure, neighborhood_search = container

    perm = sortperm(eachparticle(container),
                    by=(i -> cell_z_index(extract_svector(coordinates, container, i),
                                          neighborhood_search)))

    permute!(mass, perm)
    permute!(pressure, perm)
    Base.permutecols!!(u, perm)

    return nothing
end

@inline function cell_z_index(coords, neighborhood_search)
    cell = cell_coords(coords, neighborhood_search) .+ 1

    return cartesian2morton(SVector(cell))
end

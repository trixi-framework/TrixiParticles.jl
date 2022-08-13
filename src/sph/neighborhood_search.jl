@inline initialize!(neighborhood_search, u, semi) = nothing
@inline update!(neighborhood_search, u, semi) = nothing

# Without neighborhood search iterate over all particles
@inline eachneighbor(particle, u, neighborhood_search::Nothing, semi) = eachparticle(semi)


struct SpatialHashingSearch{NDIMS, ELTYPE}
    hashtable       ::Dict{NTuple{NDIMS, Int64}, Vector{Int64}}
    search_distance ::ELTYPE
    empty_vector    ::Vector{Int64} # Just an empty vector

    function SpatialHashingSearch{NDIMS}(search_distance) where NDIMS
        hashtable = Dict{NTuple{NDIMS, Int64}, Vector{Int64}}()
        empty_vector = Vector{Int64}()

        new{NDIMS, typeof(search_distance)}(hashtable, search_distance, empty_vector)
    end
end


function initialize!(neighborhood_search::SpatialHashingSearch, u, semi)
    @unpack hashtable = neighborhood_search

    empty!(hashtable)

    # This is needed to prevent lagging on macOS ARM.
    # See https://github.com/JuliaSIMD/Polyester.jl/issues/89
    ThreadingUtilities.sleep_all_tasks()

    for particle in eachparticle(semi)
        cell_coords = get_cell_coords(u, neighborhood_search, semi, particle)

        if haskey(hashtable, cell_coords)
            append!(hashtable[cell_coords], particle)
        else
            hashtable[cell_coords] = [particle]
        end
    end

    return neighborhood_search
end


function update!(neighborhood_search::SpatialHashingSearch, u, semi)
    @unpack hashtable = neighborhood_search

    # This is needed to prevent lagging on macOS ARM.
    # See https://github.com/JuliaSIMD/Polyester.jl/issues/89
    ThreadingUtilities.sleep_all_tasks()

    for (cell_coords, particles) in hashtable
        # Find all particles whose coordinates do not match this cell
        moved_particle_indices = (i for i in eachindex(particles)
                                  if get_cell_coords(u, neighborhood_search, semi, particles[i]) != cell_coords)

        # Add moved particles to new cell
        for i in moved_particle_indices
            particle = particles[i]
            new_cell_coords = get_cell_coords(u, neighborhood_search, semi, particle)

            if haskey(hashtable, new_cell_coords)
                append!(hashtable[new_cell_coords], particle)
            else
                hashtable[new_cell_coords] = [particle]
            end
        end

        if count(_ -> true, moved_particle_indices) == length(particles)
            # Delete this cell if all particles moved out of it
            delete!(hashtable, cell_coords)
        else
            # Remove moved particles from this cell
            deleteat!(particles, moved_particle_indices)
        end
    end

    return neighborhood_search
end


@inline function eachneighbor(particle, u, neighborhood_search::SpatialHashingSearch{2}, semi)
    cell_coords = get_cell_coords(u, neighborhood_search, semi, particle)
    x, y = cell_coords
    neighboring_cells = ((x + i, y + j) for i = -1:1, j = -1:1)

    Iterators.flatten(particles_in_cell(cell, neighborhood_search) for cell in neighboring_cells)
end

@inline function eachneighbor(particle, u, neighborhood_search::SpatialHashingSearch{3}, semi)
    cell_coords = get_cell_coords(u, neighborhood_search, semi, particle)
    x, y, z = cell_coords
    neighboring_cells = ((x + i, y + j, z + k) for i = -1:1, j = -1:1, k = -1:1)

    Iterators.flatten(particles_in_cell(cell, neighborhood_search) for cell in neighboring_cells)
end


@inline function particles_in_cell(cell_index, neighborhood_search)
    @unpack hashtable, empty_vector = neighborhood_search

    if haskey(hashtable, cell_index)
        return hashtable[cell_index]
    end

    # Reuse empty vector to avoid allocations
    return empty_vector
end


@inline function get_cell_coords(u, neighborhood_search, semi, particle)
    @unpack search_distance = neighborhood_search

    return Tuple(floor.(Int64, get_particle_coords(u, semi, particle) / search_distance))
end

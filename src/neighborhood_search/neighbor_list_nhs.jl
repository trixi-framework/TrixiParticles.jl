struct NeighborListNeighborhoodSearch{ELTYPE, NHS, PB}
    search_radius       :: ELTYPE
    periodic_box        :: PB
    grid_nhs            :: NHS
    neighbor_lists      :: Vector{Int}
    neighbor_list_start :: Vector{Int}

    function NeighborListNeighborhoodSearch(grid_nhs, n_particles)
        (; search_radius, periodic_box) = grid_nhs

        neighbor_lists = Int[]
        neighbor_list_start = zeros(Int, n_particles + 1)

        new{typeof(search_radius),
            typeof(grid_nhs), typeof(periodic_box)}(search_radius, periodic_box,
                                                    grid_nhs, neighbor_lists,
                                                    neighbor_list_start)
    end
end

@inline function Base.ndims(neighborhood_search::NeighborListNeighborhoodSearch)
    return ndims(neighborhood_search.grid_nhs)
end

function initialize!(search::NeighborListNeighborhoodSearch, coords, neighbor_coords)
    initialize!(search.grid_nhs, neighbor_coords)

    build_neighbor_lists!(search, coords, neighbor_coords)
end

function update!(search::NeighborListNeighborhoodSearch, coords, neighbor_coords)
    update!(search.grid_nhs, neighbor_coords)

    build_neighbor_lists!(search, coords, neighbor_coords)
end

@inline function eachneighbor(particle, search::NeighborListNeighborhoodSearch)
    (; neighbor_lists, neighbor_list_start) = search

    return (neighbor_lists[i] for i in neighbor_list_start[particle]:(neighbor_list_start[particle + 1] - 1))
end

function build_neighbor_lists!(search::NeighborListNeighborhoodSearch, coords,
                               neighbor_coords)
    (; grid_nhs, neighbor_lists, neighbor_list_start, search_radius, periodic_box) = search

    resize!(neighbor_lists, 0)

    for particle in 1:(length(neighbor_list_start) - 1)
        neighbor_list_start[particle] = length(neighbor_lists) + 1

        particle_coords = extract_svector(coords, Val(ndims(search)), particle)

        for neighbor in eachneighbor(particle_coords, grid_nhs)
            #     neighbor_particle_coords = extract_svector(neighbor_coords,
            #                                                Val(ndims(search)), neighbor)

            #     pos_diff = particle_coords - neighbor_particle_coords
            #     distance2 = dot(pos_diff, pos_diff)

            #     pos_diff, distance2 = compute_periodic_distance(pos_diff, distance2,
            #                                                     search_radius, periodic_box)

            #     if distance2 <= search_radius^2
            append!(neighbor_lists, neighbor)
            # end
        end
    end

    neighbor_list_start[end] = length(neighbor_lists) + 1

    return search
end

@inline function for_particle_neighbor_inner(f, system_coords, neighbor_system_coords,
                                             neighborhood_search::NeighborListNeighborhoodSearch,
                                             particle)
    (; search_radius, periodic_box) = neighborhood_search

    for neighbor in eachneighbor(particle, neighborhood_search)
        particle_coords = extract_svector(system_coords, Val(ndims(neighborhood_search)),
                                          particle)
        neighbor_coords = extract_svector(neighbor_system_coords,
                                          Val(ndims(neighborhood_search)), neighbor)

        pos_diff = particle_coords - neighbor_coords
        distance2 = dot(pos_diff, pos_diff)

        pos_diff, distance2 = compute_periodic_distance(pos_diff, distance2, search_radius,
                                                        periodic_box)

        if distance2 <= search_radius^2
            distance = sqrt(distance2)

            # Inline to avoid loss of performance
            # compared to not using `for_particle_neighbor`.
            @inline f(particle, neighbor, pos_diff, distance)
        end
    end
end

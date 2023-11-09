struct NeighborListNeighborhoodSearch{ELTYPE, NHS, PB}
    search_radius  :: ELTYPE
    periodic_box   :: PB
    grid_nhs       :: NHS
    neighbor_lists :: Vector{Vector{Int}}

    function NeighborListNeighborhoodSearch(grid_nhs, n_particles)
        (; search_radius, periodic_box) = grid_nhs

        neighbor_lists = Vector{Vector{Int}}(undef, n_particles)

        new{typeof(search_radius),
            typeof(grid_nhs), typeof(periodic_box)}(search_radius, periodic_box,
                                                    grid_nhs, neighbor_lists)
    end
end

@inline function Base.ndims(neighborhood_search::NeighborListNeighborhoodSearch)
    return ndims(neighborhood_search.grid_nhs)
end

function initialize!(search::NeighborListNeighborhoodSearch, coords, neighbor_coords)
    initialize!(search.grid_nhs, neighbor_coords)

    build_neighbor_lists!(search, coords)
end

function update!(search::NeighborListNeighborhoodSearch, coords, neighbor_coords)
    update!(search.grid_nhs, neighbor_coords)

    build_neighbor_lists!(search, coords)
end

@inline function eachneighbor(particle, particle_coords, search::NeighborListNeighborhoodSearch)
    return search.neighbor_lists[particle]
end

function build_neighbor_lists!(search::NeighborListNeighborhoodSearch, coords)
    (; grid_nhs, neighbor_lists) = search

    for particle in eachindex(neighbor_lists)
        particle_coords = extract_svector(coords, Val(ndims(search)), particle)
        neighbor_lists[particle] = collect(eachneighbor(particle_coords, grid_nhs))
    end

    return search
end

@inline function for_particle_neighbor_inner(f, system_coords, neighbor_system_coords,
                                             neighborhood_search::NeighborListNeighborhoodSearch,
                                             particle)
    (; search_radius, periodic_box) = neighborhood_search

    particle_coords = extract_svector(system_coords, Val(ndims(neighborhood_search)),
                                      particle)

    for neighbor in eachneighbor(particle, particle_coords, neighborhood_search)
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

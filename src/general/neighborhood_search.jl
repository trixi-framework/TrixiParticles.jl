# Loop over all pairs of particles and neighbors within the kernel cutoff.
# `f(particle, neighbor, pos_diff, distance)` is called for every particle-neighbor pair.
# By default, loop over `each_moving_particle(system)`.
function TrixiNeighborhoodSearch.for_particle_neighbor(f, system, neighbor_system,
                                                       system_coords, neighbor_coords,
                                                       neighborhood_search;
                                                       particles=each_moving_particle(system),
                                                       parallel=true)
    for_particle_neighbor(f, system_coords, neighbor_coords, neighborhood_search,
                          particles=particles, parallel=parallel)
end

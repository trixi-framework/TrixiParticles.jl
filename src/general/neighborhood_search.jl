# Loop over all pairs of particles and neighbors within the kernel cutoff.
# `f(particle, neighbor, pos_diff, distance)` is called for every particle-neighbor pair.
# By default, loop over `each_moving_particle(system)`.
function PointNeighbors.foreach_point_neighbor(f, system, neighbor_system,
                                               system_coords, neighbor_coords,
                                               neighborhood_search;
                                               particles=each_moving_particle(system),
                                               parallel=true)
    foreach_point_neighbor(f, system_coords, neighbor_coords, neighborhood_search,
                           particles=particles, parallel=parallel)
end

function PointNeighbors.foreach_point_neighbor(f, system::GPUSystem, neighbor_system,
                                               system_coords, neighbor_coords,
                                               neighborhood_search;
                                               particles=each_moving_particle(system),
                                               parallel=true)
    @threaded system for particle in particles
        PointNeighbors.foreach_neighbor(f, system_coords, neighbor_coords,
                                        neighborhood_search, particle)
    end
end

# Loop over all pairs of particles and neighbors within the kernel cutoff.
# `f(particle, neighbor, pos_diff, distance)` is called for every particle-neighbor pair.
# By default, loop over `eachparticle(system)`.
function PointNeighbors.foreach_point_neighbor(f, system, neighbor_system,
                                               system_coords, neighbor_coords, semi;
                                               points=eachparticle(system),
                                               parallelization_backend=semi.parallelization_backend)
    neighborhood_search = get_neighborhood_search(system, neighbor_system, semi)
    foreach_point_neighbor(f, system_coords, neighbor_coords, neighborhood_search;
                           points, parallel=parallelization_backend)
end

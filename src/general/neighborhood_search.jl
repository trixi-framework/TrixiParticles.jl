# Loop over all pairs of particles and neighbors within the kernel cutoff.
# `f(particle, neighbor, pos_diff, distance)` is called for every particle-neighbor pair.
# By default, loop over `eachparticle(system)`.
function PointNeighbors.foreach_point_neighbor(
        f, system, neighbor_system,
        system_coords, neighbor_coords, semi;
        points = eachparticle(system),
        parallelization_backend = semi.parallelization_backend
    )
    neighborhood_search = get_neighborhood_search(system, neighbor_system, semi)
    return foreach_point_neighbor(
        f, system_coords, neighbor_coords, neighborhood_search;
        points, parallelization_backend
    )
end

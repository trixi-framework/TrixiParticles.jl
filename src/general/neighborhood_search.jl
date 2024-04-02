# Loop over all pairs of particles and neighbors within the kernel cutoff.
# `f(particle, neighbor, pos_diff, distance)` is called for every particle-neighbor pair.
# By default, loop over `each_moving_particle(system)`.
function PointNeighbors.for_particle_neighbor(f, system, neighbor_system,
                                              system_coords, neighbor_coords,
                                              neighborhood_search;
                                              particles=each_moving_particle(system),
                                              parallel=true)
    for_particle_neighbor(f, system_coords, neighbor_coords, neighborhood_search,
                          particles=particles, parallel=parallel)
end

# GPU kernel version of the `@threaded` function above
function PointNeighbors.for_particle_neighbor(f, system::GPUSystem, neighbor_system,
                                              system_coords, neighbor_coords,
                                              neighborhood_search;
                                              particles=each_moving_particle(system),
                                              parallel=true)
    backend = get_backend(system_coords)

    kernel = for_particle_neighbor_kernel(backend)
    kernel(f, system_coords, neighbor_coords, neighborhood_search,
           ndrange=length(particles))

    synchronize(backend)
end

@kernel function for_particle_neighbor_kernel(f, system_coords, neighbor_coords,
                                              neighborhood_search)
    particle = @index(Global)
    PointNeighbors.for_particle_neighbor_inner(f, system_coords, neighbor_coords,
                                               neighborhood_search, particle)
end

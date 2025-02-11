# Loop over all pairs of particles and neighbors within the kernel cutoff.
# `f(particle, neighbor, pos_diff, distance)` is called for every particle-neighbor pair.
# By default, loop over `eachparticle(system)`.
function PointNeighbors.foreach_point_neighbor(f, system, neighbor_system,
                                               system_coords, neighbor_coords,
                                               neighborhood_search;
                                               points=eachparticle(system),
                                               parallel=true)
    foreach_point_neighbor(f, system_coords, neighbor_coords, neighborhood_search;
                           points, parallel)
end

function PointNeighbors.foreach_point_neighbor(f, system::GPUSystem, neighbor_system,
                                               system_coords, neighbor_coords,
                                               neighborhood_search;
                                               points=eachparticle(system),
                                               parallel=true)
    # For `GPUSystem`s, explicitly pass the backend, so a `GPUSystem` with a CPU
    # backend will actually launch the KernelAbstractions.jl kernels on the CPU.
    foreach_point_neighbor(f, system_coords, neighbor_coords, neighborhood_search;
                           points, parallel=KernelAbstractions.get_backend(system_coords))
end

# Loop over all pairs of particles and neighbors within the kernel cutoff.
# `f(particle, neighbor, pos_diff, distance)` is called for every particle-neighbor pair.
# By default, loop over `eachparticle(system)`.
function PointNeighbors.foreach_point_neighbor(f, system, neighbor_system,
                                               system_coords, neighbor_coords, semi;
                                               points=eachparticle(system),
                                               parallelization_backend=semi.parallelization_backend)
    neighborhood_search = get_neighborhood_search(system, neighbor_system, semi)
    foreach_point_neighbor(f, system_coords, neighbor_coords, neighborhood_search;
                           points, parallelization_backend)
end

deactivate_out_of_bounds_particles!(system, ::Nothing, nhs, u, semi) = system

function deactivate_out_of_bounds_particles!(system, ::SystemBuffer,
                                             cell_list::FullGridCellList, u, semi)
    (; min_corner, max_corner) = cell_list

    @threaded semi for particle in each_integrated_particle(system)
        particle_position = current_coords(u, system, particle)

        if !all(min_corner .<= particle_position .<= max_corner)
            deactivate_particle!(system, particle, u)
        end
    end

    update_system_buffer!(system.buffer, semi)

    return system
end

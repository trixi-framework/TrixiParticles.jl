# Calculate the interaction forces between particles in a Discrete Element Method (DEM) system.
#
# This function loops over all pairs of particles and their neighbors within a set distance.
# When particles overlap (i.e., they come into contact), a normal force is applied to resolve the overlap.
# This force is computed based on Hertzian contact mechanics typical for DEM simulations.
# The force is proportional to the amount of overlap and is directed along the normal between the particle centers.
# The magnitude of the force is determined by the stiffness constant `kn` and the overlap distan
function interact!(dv, v_particle_system, u_particle_system, v_neighbor_system,
                   u_neighbor_system, neighborhood_search, particle_system::DEMSystem,
                   neighbor_system::DEMSystem)
    (; mass, radius, kn) = particle_system

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    for_particle_neighbor(particle_system, neighbor_system, system_coords, neighbor_coords,
                          neighborhood_search) do particle, neighbor, pos_diff, distance
        # Only consider particles with a distance > 0.
        distance < sqrt(eps()) && return

        # Calculate the overlap (penetration depth) between the two particles
        overlap = radius[particle] + radius[neighbor] - distance

        # If there's no overlap, no force needs to be applied
        overlap <= 0 && return

        # Normal direction from particle to neighbor
        normal = pos_diff / distance

        # Compute the force magnitude using Hertzian contact mechanics
        force_magnitude = kn * overlap

        force = force_magnitude * normal

        # Update the acceleration of the particle based on the force and its mass
        @inbounds for i in 1:ndims(particle_system)
            dv[i, particle] += force[i] / mass[particle]
        end
    end

    return dv
end

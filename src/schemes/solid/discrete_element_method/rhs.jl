# Calculate the interaction forces between particles in a Discrete Element Method (DEM) system.
#
# This function loops over all pairs of particles and their neighbors within a set distance.
# When particles overlap (i.e., they come into contact), a normal force is applied to resolve the overlap.
# This force is computed based on Hertzian contact mechanics typical for DEM simulations.
# The force is proportional to the amount of overlap and is directed along the normal between the particle centers.
# The magnitude of the force is determined by the stiffness constant `kn` and the overlap distance.
function interact!(dv, v_particle_system, u_particle_system, v_neighbor_system,
                   u_neighbor_system, neighborhood_search, particle_system::DEMSystem,
                   neighbor_system)
    (; mass, radius, elastic_modulus, poissons_ratio, damping_coefficient) = particle_system

    nghb_radius = neighbor_system.radius
    nghb_mass = neighbor_system.mass
    nghb_elastic_modulus = neighbor_system.elastic_modulus
    nghb_poissons_ratio = neighbor_system.poissons_ratio

    # Compute effective modulus for both systems
    E_star = 1 / ((1 - poissons_ratio^2) / elastic_modulus + (1 - nghb_poissons_ratio^2) / nghb_elastic_modulus)

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    for_particle_neighbor(particle_system, neighbor_system, system_coords, neighbor_coords,
                          neighborhood_search) do particle, neighbor, pos_diff, distance
        # Only consider particles with a distance > 0.
        distance < sqrt(eps()) && return

        # Calculate the overlap (penetration depth) between the two particles
        overlap = radius[particle] + nghb_radius[neighbor] - distance

        # If there's no overlap, no force needs to be applied
        overlap <= 0 && return

        # Compute effective radius for the interaction
        r_star = (radius[particle] * nghb_radius[neighbor]) /
                 (radius[particle] + nghb_radius[neighbor])

        # Compute stiffness constant kn for the interaction
        kn = (4 / 3) * E_star * sqrt(r_star * overlap)

        # Calculate effective mass for the interaction
        m_star = (mass[particle] * nghb_mass[neighbor]) /
                 (mass[particle] + nghb_mass[neighbor])

        # Calculate critical damping coefficient
        gamma_c = 2 * sqrt(m_star * kn)

        # Normal direction from particle to neighbor
        normal = pos_diff / distance

        part_v = extract_svector(v_particle_system, particle_system, particle)
        nghbr_v = extract_svector(v_neighbor_system, neighbor_system, neighbor)

        # Relative velocity
        rel_vel = part_v - nghbr_v
        rel_vel_normal = dot(rel_vel, normal)

        # Compute the force magnitude using Hertzian contact mechanics with damping
        force_magnitude = kn * overlap + damping_coefficient * gamma_c * rel_vel_normal

        force = force_magnitude * normal

        # Update the acceleration of the particle based on the force and its mass
        @inbounds for i in 1:ndims(particle_system)
            dv[i, particle] += force[i] / mass[particle]
        end
    end

    return dv
end

function interact!(dv, v_particle_system, u_particle_system, v_neighbor_system,
                   u_neighbor_system, neighborhood_search, particle_system::DEMSystem,
                   neighbor_system::BoundaryDEMSystem)
    (; mass, radius, kn) = particle_system

    nghb_kn = neighbor_system.kn
    nghb_radius = neighbor_system.radius

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    for_particle_neighbor(particle_system, neighbor_system, system_coords, neighbor_coords,
                          neighborhood_search) do particle, neighbor, pos_diff, distance
        # Only consider particles with a distance > 0.
        distance < sqrt(eps()) && return

        # Calculate the overlap (penetration depth) between the two particles
        overlap = radius[particle] + nghb_radius[neighbor] - distance

        # If there's no overlap, no force needs to be applied
        overlap <= 0 && return

        # Normal direction from particle to neighbor
        normal = pos_diff / distance

        # Position correction to prevent penetration
        position_correction_factor = 0.5 # you can tweak this value
        @inbounds for i in 1:ndims(particle_system)
            u_particle_system[i, particle] -= position_correction_factor * overlap *
                                              normal[i]
        end

        # Compute the force magnitude using Hertzian contact mechanics
        force_magnitude = nghb_kn * position_correction_factor * overlap

        force = force_magnitude * normal

        # Reset the velocity component in the normal direction if it's pushing the particle into the wall
        normal_velocity = dot(dv[:, particle], normal)
        if normal_velocity < 0
            @inbounds for i in 1:ndims(particle_system)
                dv[i, particle] -= normal_velocity * normal[i]
                # we still need to push against the particles coming to close
                dv[i, particle] += force[i] / mass[particle]
            end
        end
    end

    return dv
end

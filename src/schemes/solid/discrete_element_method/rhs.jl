# Calculate the interaction forces between particles in a Discrete Element Method (DEM) system.
#
# This function loops over all pairs of particles and their neighbors within a set distance.
# When particles overlap (i.e., they come into contact), a normal force is applied to resolve the overlap.
# This force is computed based on Hertzian contact mechanics typical for DEM simulations.
# The force is proportional to the amount of overlap and is directed along the normal between the particle centers.
# The magnitude of the force is determined by the stiffness constant `normal_stiffness` and the overlap distance.
function interact!(dv, v_particle_system, u_particle_system, v_neighbor_system,
                   u_neighbor_system, neighborhood_search, particle_system::DEMSystem,
                   neighbor_system::Union{BoundaryDEMSystem, DEMSystem})
    (; damping_coefficient) = particle_system

    E_a = particle_system.elastic_modulus
    nu_a = particle_system.poissons_ratio

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    for_particle_neighbor(particle_system, neighbor_system, system_coords, neighbor_coords,
                          neighborhood_search) do particle, neighbor, pos_diff, distance
        m_a = particle_system.mass[particle]

        r_a = particle_system.radius[particle]
        r_b = neighbor_system.radius[neighbor]

        # Only consider particles with a distance > 0
        distance < sqrt(eps()) && return

        # Calculate the overlap (penetration depth) between the two particles
        overlap = r_a + r_b - distance

        # If there's no overlap, no force needs to be applied
        overlap <= 0 && return

        # Normal direction from neighbor to particle
        normal = pos_diff / distance

        interaction_force = collision_force(particle_system, neighbor_system, overlap,
                                            normal, v_particle_system,
                                            v_neighbor_system, E_a,
                                            nu_a, r_a, r_b, m_a,
                                            damping_coefficient, particle, neighbor)

        # Update the acceleration of the particle based on the force and its mass
        for i in 1:ndims(particle_system)
            dv[i, particle] += interaction_force[i] / m_a
        end

        # TODO: use update callback
        position_correction!(neighbor_system, u_particle_system, overlap, normal, particle)
    end

    return dv
end

@inline function collision_force(particle_system, neighbor_system::BoundaryDEMSystem,
                                 overlap, normal, v_particle_system,
                                 v_neighbor_system, E_a, nu_a,
                                 r_a, r_b, m_a, damping_coefficient,
                                 particle, neighbor)
    return neighbor_system.normal_stiffness * overlap * normal
end

@inline function collision_force(particle_system, neighbor_system::DEMSystem, overlap,
                                 normal, v_particle_system, v_neighbor_system,
                                 E_a, nu_a, r_a, r_b,
                                 m_a, damping_coefficient, particle, neighbor)
    m_b = neighbor_system.mass[neighbor]
    E_b = neighbor_system.elastic_modulus
    nu_b = neighbor_system.poissons_ratio

    v_a = current_velocity(v_particle_system, particle_system, particle)
    v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)

    v_ab = v_a - v_b
    rel_vel_normal = dot(v_ab, normal)

    # Compute effective modulus for both systems
    E_star = 1 / ((1 - nu_a^2) / E_a + (1 - nu_b^2) / E_b)

    # Compute effective radius for the interaction
    r_star = (r_a * r_b) / (r_a + r_b)

    # Compute stiffness constant normal_stiffness for the interaction
    normal_stiffness = (4 / 3) * E_star * sqrt(r_star * overlap)

    # Calculate effective mass for the interaction
    m_star = (m_a * m_b) / (m_a + m_b)

    # Calculate critical damping coefficient
    gamma_c = 2 * sqrt(m_star * normal_stiffness)

    # Compute the force magnitude using Hertzian contact mechanics with damping
    force_magnitude = normal_stiffness * overlap +
                      damping_coefficient * gamma_c * rel_vel_normal

    return force_magnitude * normal
end

@inline function position_correction!(neighbor_system, u_particle_system, overlap, normal,
                                      particle)
end

@inline function position_correction!(neighbor_system::BoundaryDEMSystem, u_particle_system,
                                      overlap, normal, particle)
    for i in 1:ndims(neighbor_system)
        # Position correction to prevent penetration
        u_particle_system[i, particle] -= 0.5 * overlap * normal[i]
    end
end

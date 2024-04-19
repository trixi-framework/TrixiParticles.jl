# Calculate the interaction forces between particles in a Discrete Element Method (DEM) system.
#
# This function loops over all pairs of particles and their neighbors within a set distance.
# When particles overlap (i.e., they come into contact), a normal force is applied to resolve the overlap.
# This force is computed based on Hertzian contact mechanics typical for DEM simulations.
# The force is proportional to the amount of overlap and is directed along the normal between the particle centers.
# The magnitude of the force is determined by the stiffness constant `kn` and the overlap distance.
function interact!(dv, v_particle_system, u_particle_system, v_neighbor_system,
                   u_neighbor_system, neighborhood_search, particle_system::DEMSystem,
                   neighbor_system::Union{BoundaryDEMSystem, DEMSystem})
    (; mass, radius, elastic_modulus, poissons_ratio, damping_coefficient) = particle_system

    m_a = mass
    r_a = radius
    E_a = elastic_modulus
    nu_a = poissons_ratio

    r_b = neighbor_system.radius

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    for_particle_neighbor(particle_system, neighbor_system, system_coords, neighbor_coords,
                          neighborhood_search) do particle, neighbor, pos_diff, distance
        # Only consider particles with a distance > 0.
        distance < sqrt(eps()) && return

        # Calculate the overlap (penetration depth) between the two particles
        overlap = r_a[particle] + r_b[neighbor] - distance

        # If there's no overlap, no force needs to be applied
        overlap <= 0 && return

        # Normal direction from particle to neighbor
        normal = pos_diff / distance

        interaction_force = collision_interaction(particle_system, neighbor_system, overlap,
                                                  normal, v_particle_system,
                                                  v_neighbor_system, E_a,
                                                  nu_a, r_a, r_b, m_a,
                                                  damping_coefficient, particle, neighbor)

        # Update the acceleration of the particle based on the force and its mass
        for i in 1:ndims(particle_system)
            dv[i, particle] += interaction_force[i] / m_a[particle]
        end

        position_correction!(neighbor_system, u_particle_system, overlap, normal, particle)
    end

    return dv
end

@inline function collision_interaction(particle_system, neighbor_system::BoundaryDEMSystem,
                                       overlap, normal, v_particle_system,
                                       v_neighbor_system, E_a, nu_a,
                                       r_a, r_b, m_a, damping_coefficient,
                                       particle, neighbor)
    return neighbor_system.kn * overlap * normal
end

@inline function collision_interaction(particle_system, neighbor_system::DEMSystem, overlap,
                                       normal, v_particle_system, v_neighbor_system,
                                       E_a, nu_a, r_a, r_b,
                                       m_a, damping_coefficient, particle, neighbor)
    v_a = extract_svector(v_particle_system, particle_system, particle)
    v_b = extract_svector(v_neighbor_system, neighbor_system, neighbor)

    relative_velocity = v_a - v_b
    rel_vel_normal = dot(relative_velocity, normal)

    # Compute effective modulus for both systems
    E_star = 1 / ((1 - nu_a^2) / E_a +
              (1 - neighbor_system.poissons_ratio^2) / neighbor_system.elastic_modulus)

    # Compute effective radius for the interaction
    r_star = (r_a[particle] * r_b[neighbor]) /
             (r_a[particle] + r_b[neighbor])

    # Compute stiffness constant kn for the interaction
    kn = (4 / 3) * E_star * sqrt(r_star * overlap)

    # Calculate effective mass for the interaction
    m_star = (m_a[particle] * neighbor_system.mass[neighbor]) /
             (m_a[particle] + neighbor_system.mass[neighbor])

    # Calculate critical damping coefficient
    gamma_c = 2 * sqrt(m_star * kn)

    # Compute the force magnitude using Hertzian contact mechanics with damping
    force_magnitude = kn * overlap + damping_coefficient * gamma_c * rel_vel_normal

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

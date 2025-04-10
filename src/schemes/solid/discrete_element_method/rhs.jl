function interact!(dv, v_particle_system, u_particle_system, v_neighbor_system,
                   u_neighbor_system, particle_system::DEMSystem,
                   neighbor_system::Union{BoundaryDEMSystem, DEMSystem}, semi)
    damping_coefficient = particle_system.damping_coefficient

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    foreach_point_neighbor(particle_system, neighbor_system, system_coords, neighbor_coords,
                           semi;
                           points=each_moving_particle(particle_system)) do particle,
                                                                            neighbor,
                                                                            pos_diff,
                                                                            distance
        distance < sqrt(eps()) && return

        # Retrieve particle properties.
        m_a = particle_system.mass[particle]
        r_a = particle_system.radius[particle]
        r_b = neighbor_system.radius[neighbor]

        # Compute the overlap (penetration depth).
        overlap = r_a + r_b - distance
        if overlap > 0
            # Compute the unit normal vector (from neighbor to particle).
            normal = pos_diff / distance

            # Compute Normal Force by Dispatching on the Contact Model.
            F_normal = collision_force_normal(particle_system.contact_model,
                                              particle_system, neighbor_system,
                                              overlap, normal,
                                              v_particle_system, v_neighbor_system,
                                              particle, neighbor, damping_coefficient)

            F_tangent = collision_force_tangential(particle_system, neighbor_system,
                                                   overlap, normal, v_particle_system,
                                                   v_neighbor_system,
                                                   particle, neighbor, F_normal)

            interaction_force = F_normal + F_tangent

            # Update the particle acceleration: a = F/m.
            for i in 1:ndims(particle_system)
                dv[i, particle] += interaction_force[i] / m_a
            end

            # Apply a simple position correction to mitigate overlap.
            position_correction!(neighbor_system, u_particle_system, overlap, normal,
                                 particle)
        end
    end

    return dv
end

# Tangential Force Computation (common for now)
#
# Uses a spring-dashpot model to compute the instantaneous tangential force,
# with a Coulomb friction limit.
@inline function collision_force_tangential(particle_system, neighbor_system,
                                            overlap, normal,
                                            v_particle_system, v_neighbor_system,
                                            particle, neighbor, F_normal)
    # Tangential force parameters
    friction_coefficient = 0.5       # Coulomb friction coefficient [Cundall and Strack, 1979]
    tangential_stiffness = 1e3       # Tangential spring constant
    tangential_damping = 0.001       # Damping coefficient for tangential force

    # Compute relative velocity and extract the tangential component.
    v_a = current_velocity(v_particle_system, particle_system, particle)
    v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)
    v_rel = v_a - v_b
    v_rel_tangent = v_rel - (dot(v_rel, normal) * normal)

    # Compute tangential force as a spring–dashpot response.
    F_t = -tangential_stiffness * v_rel_tangent - tangential_damping * v_rel_tangent

    # Coulomb friction: limit the tangential force to μ * |F_normal|.
    max_tangent = friction_coefficient * norm(F_normal)
    if norm(F_t) > max_tangent && norm(F_t) > 0.0
        F_t = F_t * (max_tangent / norm(F_t))
    end

    return F_t
end

@inline function position_correction!(neighbor_system::DEMSystem,
                                      u_particle_system, overlap, normal, particle)
end

# For boundaries, a simple half-overlap correction is applied.
@inline function position_correction!(neighbor_system::BoundaryDEMSystem,
                                      u_particle_system, overlap, normal, particle)
    for i in 1:ndims(neighbor_system)
        u_particle_system[i, particle] -= 0.5 * overlap * normal[i]
    end
end

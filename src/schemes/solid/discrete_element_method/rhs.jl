function interact!(dv, v_particle_system, u_particle_system, v_neighbor_system,
                   u_neighbor_system, neighborhood_search, particle_system::DEMSystem,
                   neighbor_system::Union{BoundaryDEMSystem, DEMSystem})
    # Extract global parameters.
    damping_coefficient = particle_system.damping_coefficient

    # The contact model is stored within the DEM system.
    cm = particle_system.contact_model

    # Tangential force parameters (could be made part of a TangentialModel type too)
    friction_coefficient = 0.5       # Coulomb friction coefficient [Cundall and Strack, 1979]
    tangential_stiffness = 1e3       # Tangential spring constant
    tangential_damping = 0.001     # Damping coefficient for tangential force

    # Get current coordinates.
    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    foreach_point_neighbor(particle_system, neighbor_system, system_coords, neighbor_coords,
                           neighborhood_search;
                           points=each_moving_particle(particle_system)) do particle,
                                                                            neighbor,
                                                                            pos_diff,
                                                                            distance

        # Retrieve particle properties.
        m_a = particle_system.mass[particle]
        r_a = particle_system.radius[particle]
        r_b = neighbor_system.radius[neighbor]

        # Avoid division by zero for nearly coincident particles.
        if distance < sqrt(eps())
            continue
        end

        # Compute the overlap (penetration depth).
        overlap = r_a + r_b - distance
        if overlap <= 0
            continue  # No contact: no force to compute.
        end

        # Compute the unit normal vector (from neighbor to particle).
        normal = pos_diff / distance

        #-------------------------------------------------------------------------
        # Compute Normal Force by Dispatching on the Contact Model
        #-------------------------------------------------------------------------
        F_normal = collision_force_normal(cm, particle_system, neighbor_system, overlap,
                                          normal,
                                          v_particle_system, v_neighbor_system, particle,
                                          neighbor,
                                          damping_coefficient)

        #-------------------------------------------------------------------------
        # Compute Tangential Force (spring–dashpot model with Coulomb friction)
        #-------------------------------------------------------------------------
        F_tangent = collision_force_tangential(particle_system, neighbor_system, overlap,
                                               normal,
                                               v_particle_system, v_neighbor_system,
                                               particle, neighbor,
                                               tangential_stiffness, tangential_damping,
                                               friction_coefficient)

        # Sum the normal and tangential forces.
        interaction_force = F_normal + F_tangent

        # Update the particle acceleration: a = F/m.
        for i in 1:ndims(particle_system)
            dv[i, particle] += interaction_force[i] / m_a
        end

        # Apply a simple position correction to mitigate overlap.
        position_correction!(neighbor_system, u_particle_system, overlap, normal, particle)
    end

    return dv
end

#-------------------------------------------------------------------------
# collision_force_normal for HertzContactModel
#
# Implements a Hertzian contact law following [Di Renzo and Di Maio, 2004].
#-------------------------------------------------------------------------
@inline function collision_force_normal(model::HertzContactModel,
                                        particle_system, neighbor_system,
                                        overlap, normal, v_particle_system,
                                        v_neighbor_system,
                                        particle, neighbor, damping_coefficient)
    # Use material properties from the Hertz contact model.
    E_a = model.elastic_modulus
    nu_a = model.poissons_ratio

    # For the neighbor, use its properties if available (otherwise, assume identical material).
    if neighbor_system isa DEMSystem
        E_b = neighbor_system.elastic_modulus
        nu_b = neighbor_system.poissons_ratio
    else
        E_b = E_a
        nu_b = nu_a
    end

    # Compute relative velocity along the contact normal.
    v_a = current_velocity(v_particle_system, particle_system, particle)
    v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)
    rel_vel = v_a - v_b
    rel_vel_norm = dot(rel_vel, normal)

    # Compute the effective elastic modulus (Hertzian theory).
    E_star = 1 / (((1 - nu_a^2) / E_a) + ((1 - nu_b^2) / E_b))

    # Compute the effective radius.
    r_a = particle_system.radius[particle]
    r_b = neighbor_system.radius[neighbor]
    r_star = (r_a * r_b) / (r_a + r_b)

    # Non-linear stiffness for Hertzian contact.
    normal_stiffness = (4 / 3) * E_star * sqrt(r_star * overlap)

    # Compute effective mass for damping.
    if neighbor_system isa DEMSystem
        m_a = particle_system.mass[particle]
        m_b = neighbor_system.mass[neighbor]
        m_star = (m_a * m_b) / (m_a + m_b)
    else
        m_star = particle_system.mass[particle]
    end

    # Critical damping coefficient.
    gamma_c = 2 * sqrt(m_star * normal_stiffness)

    # Total normal force: elastic + damping term.
    force_magnitude = normal_stiffness * overlap +
                      damping_coefficient * gamma_c * rel_vel_norm

    return force_magnitude * normal
end

#-------------------------------------------------------------------------
# collision_force_normal for LinearContactModel
#
# Implements a linear spring-dashpot contact model [Cundall and Strack, 1979].
#-------------------------------------------------------------------------
@inline function collision_force_normal(model::LinearContactModel,
                                        particle_system, neighbor_system,
                                        overlap, normal, v_particle_system,
                                        v_neighbor_system,
                                        particle, neighbor, damping_coefficient)
    # Use the constant stiffness from the linear contact model.
    normal_stiffness = model.normal_stiffness

    v_a = current_velocity(v_particle_system, particle_system, particle)
    v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)
    rel_vel = v_a - v_b
    rel_vel_norm = dot(rel_vel, normal)

    # Compute effective mass for damping.
    if neighbor_system isa DEMSystem
        m_a = particle_system.mass[particle]
        m_b = neighbor_system.mass[neighbor]
        m_star = (m_a * m_b) / (m_a + m_b)
    else
        m_star = particle_system.mass[particle]
    end

    gamma_c = 2 * sqrt(m_star * normal_stiffness)
    force_magnitude = normal_stiffness * overlap +
                      damping_coefficient * gamma_c * rel_vel_norm

    return force_magnitude * normal
end

#-------------------------------------------------------------------------
# Tangential Force Computation (common for now)
#
# Uses a spring-dashpot model to compute the instantaneous tangential force,
# with a Coulomb friction limit.
#-------------------------------------------------------------------------
@inline function collision_force_tangential(particle_system, neighbor_system,
                                            overlap, normal,
                                            v_particle_system, v_neighbor_system,
                                            particle, neighbor,
                                            tangential_stiffness, tangential_damping,
                                            friction_coefficient)
    # Compute relative velocity and extract tangential component.
    v_a = current_velocity(v_particle_system, particle_system, particle)
    v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)
    v_rel = v_a - v_b
    v_rel_tangent = v_rel - dot(v_rel, normal) * normal

    # Compute tangential force as spring–dashpot response.
    F_t = -tangential_stiffness * v_rel_tangent - tangential_damping * v_rel_tangent

    # Coulomb friction: limit the tangential force to μ * |F_normal|.
    # Here, we estimate F_normal as (normal_stiffness * overlap) using the base stiffness.
    F_n_est = particle_system.normal_stiffness * overlap
    max_tangent = friction_coefficient * abs(F_n_est)
    if norm(F_t) > max_tangent && norm(F_t) > 0.0
        F_t = F_t * (max_tangent / norm(F_t))
    end

    return F_t
end

#-------------------------------------------------------------------------
# position_correction!: Adjusts particle positions to mitigate excessive overlap.
#
# For boundaries, a simple half-overlap correction is applied.
#-------------------------------------------------------------------------
@inline function position_correction!(neighbor_system::BoundaryDEMSystem,
                                      u_particle_system, overlap, normal, particle)
    for i in 1:ndims(neighbor_system)
        u_particle_system[i, particle] -= 0.5 * overlap * normal[i]
    end
end

# Optionally, provide an empty version for DEM–DEM interactions if corrections
# are handled by the integration scheme.
@inline function position_correction!(neighbor_system::DEMSystem,
                                      u_particle_system, overlap, normal, particle)
    # DEM–DEM position correction can be implemented here if desired.
end

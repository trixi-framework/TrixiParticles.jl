# Define an abstract type for contact models.
abstract type ContactModel end

# HertzContactModel: Non-linear contact using Hertzian theory.
struct HertzContactModel <: ContactModel
    elastic_modulus::Float64  # Material elastic modulus
    poissons_ratio::Float64   # Material Poisson's ratio
end

# Normal collision force for the HertzContactModel
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

    # TODO: needs to be dispatched and handled for boundaries
    # For the neighbor, use its properties if available (otherwise, assume identical material).
    if neighbor_system isa DEMSystem && neighbor_system.contact_model isa HertzContactModel
        E_b = neighbor_system.contact_model.elastic_modulus
        nu_b = neighbor_system.contact_model.poissons_ratio
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

# LinearContactModel: Simple linear spring-dashpot contact.
struct LinearContactModel <: ContactModel
    normal_stiffness::Float64 # Constant stiffness value for linear contact
end

# Normal collision force for the LinearContactModel
#
# Implements a linear spring-dashpot contact model [Cundall and Strack, 1979].
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

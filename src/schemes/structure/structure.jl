# Shared structure-fluid interaction helpers used by multiple structure schemes.

@propagate_inbounds function structure_fluid_interaction(v_particle_system,
                                                         v_neighbor_system,
                                                         particle_system::AbstractStructureSystem,
                                                         neighbor_system::AbstractFluidSystem,
                                                         particle, neighbor,
                                                         pos_diff, distance,
                                                         sound_speed,
                                                         grad_kernel, m_b, rho_a, rho_b)
    surface_tension = surface_tension_model(neighbor_system)

    # In fluid-structure interaction, use the "hydrodynamic mass" of the structure particles
    # corresponding to the rest density of the fluid and not the material density.
    m_a = hydrodynamic_mass(particle_system, particle)

    # In fluid-structure interaction, use the "hydrodynamic pressure" of the structure particles
    # corresponding to the chosen boundary model.
    p_a = current_pressure(v_particle_system, particle_system, particle)
    p_b = current_pressure(v_neighbor_system, neighbor_system, neighbor)

    # Particle and neighbor (and the corresponding systems and particle quantities) are
    # switched in the following two calls. This yields the exact same pair force as in the
    # fluid-structure interaction, but with flipped sign because `pos_diff` is reversed.
    dv_boundary = pressure_acceleration(neighbor_system, particle_system,
                                        neighbor, particle,
                                        m_b, m_a, p_b, p_a, rho_b, rho_a,
                                        pos_diff, distance, grad_kernel,
                                        system_correction(neighbor_system))

    dv_viscosity_ = dv_viscosity(neighbor_system, particle_system,
                                 v_neighbor_system, v_particle_system,
                                 neighbor, particle, pos_diff, distance,
                                 sound_speed, m_b, m_a,
                                 rho_b, rho_a,
                                 grad_kernel)

    dv_adhesion = adhesion_force(surface_tension, neighbor_system, particle_system,
                                 neighbor, particle, pos_diff, distance)

    # Keep the shared helper acceleration-like; callers convert it to the quantity they store.
    dv_fs = dv_boundary + dv_viscosity_ + dv_adhesion

    return dv_fs
end

@inline function structure_fluid_continuity!(dv, v_particle_system, v_neighbor_system,
                                             particle, neighbor, m_b, rho_a, rho_b,
                                             particle_system::AbstractStructureSystem,
                                             neighbor_system::AbstractFluidSystem,
                                             grad_kernel)
    return dv
end

@inline function structure_fluid_continuity!(dv, v_particle_system, v_neighbor_system,
                                             particle, neighbor, m_b, rho_a, rho_b,
                                             particle_system::Union{RigidBodySystem{<:BoundaryModelDummyParticles{ContinuityDensity}},
                                                                    TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles{ContinuityDensity}}},
                                             neighbor_system::AbstractFluidSystem,
                                             grad_kernel)
    v_diff = current_velocity(v_particle_system, particle_system, particle) -
             current_velocity(v_neighbor_system, neighbor_system, neighbor)

    continuity_equation!(dv, density_calculator(neighbor_system), m_b, rho_a, rho_b, v_diff,
                         grad_kernel, particle)
end

# Physical momentum pair operators shared by fluid RHSs and structure reactions.
# Continuity, EDAC pressure evolution, and shifting transport are assembled separately.
@propagate_inbounds function add_momentum_equation(dv_particle,
                                                   particle_system::Union{WeaklyCompressibleSPHSystem,
                                                                          EntropicallyDampedSPHSystem,
                                                                          ImplicitIncompressibleSPHSystem},
                                                   neighbor_system,
                                                   v_particle_system, v_neighbor_system,
                                                   particle, neighbor, pos_diff, distance,
                                                   grad_kernel, sound_speed, m_a, m_b,
                                                   p_a, p_b, rho_a, rho_b, v_a, v_b)
    return dv_particle +
           physical_fluid_pair_acceleration(particle_system, neighbor_system,
                                            v_particle_system, v_neighbor_system,
                                            particle, neighbor, m_a, m_b, p_a, p_b,
                                            rho_a, rho_b, v_a, v_b, pos_diff, distance,
                                            sound_speed, grad_kernel,
                                            system_correction(particle_system))
end

@propagate_inbounds function neighbor_pressure(v_neighbor_system, neighbor_system,
                                               neighbor, p_a)
    return current_pressure(v_neighbor_system, neighbor_system, neighbor)
end

@inline function neighbor_pressure(v_neighbor_system,
                                   neighbor_system::Union{WallBoundarySystem{<:BoundaryModelDummyParticles{PressureMirroring}},
                                                          TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles{PressureMirroring}},
                                                          RigidBodySystem{<:BoundaryModelDummyParticles{PressureMirroring}}},
                                   neighbor, p_a)
    return p_a
end

# EDAC supplies its own particle-local pressure average. Other schemes keep their
# absolute pressures by subtracting zero in the physical pair operator.
@inline average_pressure(system, particle) = zero(eltype(system))

# EDAC/IISPH use unity; WCSPH delegates to the free-surface correction configured
# for its fluid-side pair force. The structure reaction must use the same factors.
@inline function interaction_force_correction(system, rho_a, rho_b)
    return 1, 1, 1
end

@inline function interaction_force_correction(system::WeaklyCompressibleSPHSystem,
                                              rho_a, rho_b)
    return free_surface_correction(system_correction(system), system, rho_a, rho_b)
end

@inline function sum_interaction_contributions(a, b)
    # Reduce a momentum/force vector and density-rate scalar separately, preserving
    # their distinct meaning while avoiding per-neighbor writes to the RHS arrays.
    dv_a, drho_a = a
    dv_b, drho_b = b
    return dv_a + dv_b, drho_a + drho_b
end

# Here a is the fluid particle and b its neighbor: pos_diff=x_a-x_b, with the
# gradient evaluated at a in that direction. Return acceleration of a; a structure
# caller then applies -m_a to obtain the opposite force on b.
@propagate_inbounds function physical_fluid_pair_acceleration(particle_system,
                                                              neighbor_system,
                                                              v_particle_system,
                                                              v_neighbor_system,
                                                              particle, neighbor,
                                                              m_a, m_b, p_a, p_b, rho_a,
                                                              rho_b,
                                                              v_a, v_b, pos_diff, distance,
                                                              sound_speed, grad_kernel,
                                                              correction)
    # EDAC subtracts the fluid particle's local mean from both pair pressures,
    # including a boundary pressure, so the reaction matches its reduced fluid force.
    p_avg = average_pressure(particle_system, particle)
    viscosity_correction, pressure_correction,
    surface_tension_correction = interaction_force_correction(particle_system, rho_a, rho_b)

    dv_pressure = pressure_acceleration(particle_system, neighbor_system,
                                        particle, neighbor, m_a, m_b,
                                        p_a - p_avg, p_b - p_avg, rho_a, rho_b,
                                        pos_diff, distance, grad_kernel, correction)
    dv_particle = dv_pressure * pressure_correction

    # Keep systems and kinematics fluid-first together. The helper supplies ghost
    # velocities for no-slip models and preserves artificial viscosity's approach test.
    dv_particle = add_dv_viscosity(dv_particle, particle_system, neighbor_system,
                                   v_particle_system, v_neighbor_system,
                                   particle, neighbor, pos_diff, distance, sound_speed,
                                   m_a, m_b, rho_a, rho_b, v_a, v_b, grad_kernel,
                                   viscosity_correction)

    # Surface-force dispatch skips unsupported pair models; rigid-body adhesion
    # uses this same fluid-frame displacement before the caller takes the reaction.
    surface_tension_a = surface_tension_model(particle_system)
    surface_tension_b = surface_tension_model(neighbor_system)
    dv_particle = add_dv_surface_tension(dv_particle, surface_tension_a, surface_tension_b,
                                         particle_system, neighbor_system, particle,
                                         neighbor,
                                         pos_diff, distance, rho_a, rho_b, grad_kernel,
                                         surface_tension_correction)
    dv_particle = add_dv_adhesion(dv_particle, surface_tension_a,
                                  particle_system, neighbor_system, particle, neighbor,
                                  pos_diff, distance)

    # Shifting is a transport correction, not interfacial traction. Apply its
    # momentum terms separately in the fluid RHS, not to the structural load.
    return dv_particle
end

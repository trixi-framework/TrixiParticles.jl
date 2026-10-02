# Physical momentum pair operators shared by fluid RHSs and structure reactions.
# Continuity, EDAC pressure evolution, and shifting transport are assembled separately.
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

# EDAC supplies its own average-pressure reduction; other schemes subtract zero.
@inline average_pressure(system, particle) = zero(eltype(system))

@inline function interaction_force_correction(system, rho_a, rho_b)
    return 1, 1, 1
end

@inline function interaction_force_correction(system::WeaklyCompressibleSPHSystem,
                                              rho_a, rho_b)
    return free_surface_correction(system_correction(system), system, rho_a, rho_b)
end

@inline function interaction_zero_distance(system, neighbor_system)
    return sqrt(eps(initial_smoothing_length(system)^2))
end

@inline function interaction_zero_distance(system::WeaklyCompressibleSPHSystem,
                                           neighbor_system)
    return sqrt(eps(compact_support(system, neighbor_system)^2))
end

@inline function skip_fluid_pair(system, distance, support, almostzero)
    return distance > support || (skip_zero_distance(system) && distance < almostzero)
end

@inline function sum_interaction_contributions(a, b)
    dv_a, drho_a = a
    dv_b, drho_b = b
    return dv_a + dv_b, drho_a + drho_b
end

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
    p_avg = average_pressure(particle_system, particle)
    viscosity_correction, pressure_correction,
    surface_tension_correction = interaction_force_correction(particle_system, rho_a, rho_b)

    dv_pressure = pressure_acceleration(particle_system, neighbor_system,
                                        particle, neighbor, m_a, m_b,
                                        p_a - p_avg, p_b - p_avg, rho_a, rho_b,
                                        pos_diff, distance, grad_kernel, correction)
    dv_particle = dv_pressure * pressure_correction

    dv_particle = add_dv_viscosity(dv_particle, particle_system, neighbor_system,
                                   v_particle_system, v_neighbor_system,
                                   particle, neighbor, pos_diff, distance, sound_speed,
                                   m_a, m_b, rho_a, rho_b, v_a, v_b, grad_kernel,
                                   viscosity_correction)

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

# Add the acceleration of `particle` due to `neighbor` to `dv_particle`.
# `particle` must be in `particle_system` and `neighbor` must be in `neighbor_system`.
# This includes pressure, viscosity, surface tension and adhesion, but not the extra terms
# from shifting techniques (see `interact_structure_fluid!`).
# Note that this function is also used for the structure-fluid interaction to compute
# the exact opposite pair force. When adding new terms here, make sure that they are
# also valid for structure neighbors.
@propagate_inbounds function add_momentum_equation(dv_particle,
                                                   particle_system::Union{WeaklyCompressibleSPHSystem,
                                                                          EntropicallyDampedSPHSystem,
                                                                          ImplicitIncompressibleSPHSystem},
                                                   neighbor_system,
                                                   v_particle_system, v_neighbor_system,
                                                   particle, neighbor, pos_diff, distance,
                                                   grad_kernel, sound_speed, m_a, m_b,
                                                   p_a, p_b, rho_a, rho_b, v_a, v_b)
    correction = system_correction(particle_system)
    surface_tension_a = surface_tension_model(particle_system)
    surface_tension_b = surface_tension_model(neighbor_system)

    # This technique by Basa et al. 2017 (10.1002/fld.1927) aims to reduce numerical
    # errors due to large pressures by subtracting the average pressure of neighboring
    # particles.
    # It results in significant improvement for EDAC, especially with TVF,
    # but not for WCSPH, according to Ramachandran & Puri (2019), Section 3.2.
    # Note that the return value is zero when not using average pressure reduction.
    p_avg = average_pressure(particle_system, particle)

    # WCSPH uses its free-surface correction; EDAC and IISPH keep unit factors.
    viscosity_correction, pressure_correction,
    surface_tension_correction = interaction_force_correction(particle_system, rho_a, rho_b)

    # For `ContinuityDensity` without correction or average pressure reduction,
    # this is equivalent to -m_b * (p_a + p_b) / (rho_a * rho_b) * grad_kernel.
    dv_pressure = pressure_acceleration(particle_system, neighbor_system,
                                        particle, neighbor, m_a, m_b,
                                        p_a - p_avg, p_b - p_avg, rho_a, rho_b,
                                        pos_diff, distance, grad_kernel, correction)
    dv_particle += dv_pressure * pressure_correction

    dv_particle = add_dv_viscosity(dv_particle, particle_system, neighbor_system,
                                   v_particle_system, v_neighbor_system,
                                   particle, neighbor, pos_diff, distance, sound_speed,
                                   m_a, m_b, rho_a, rho_b, v_a, v_b, grad_kernel,
                                   viscosity_correction)

    dv_particle = add_dv_surface_tension(dv_particle, surface_tension_a, surface_tension_b,
                                         particle_system, neighbor_system, particle,
                                         neighbor,
                                         pos_diff, distance, rho_a, rho_b, grad_kernel,
                                         surface_tension_correction)
    dv_particle = add_dv_adhesion(dv_particle, surface_tension_a,
                                  particle_system, neighbor_system, particle, neighbor,
                                  pos_diff, distance)

    return dv_particle
end

@inline function average_pressure(system::Union{WeaklyCompressibleSPHSystem,
                                                ImplicitIncompressibleSPHSystem}, particle)
    return zero(eltype(system))
end

@inline function interaction_force_correction(system::Union{EntropicallyDampedSPHSystem,
                                                            ImplicitIncompressibleSPHSystem},
                                              rho_a, rho_b)
    return 1, 1, 1
end

@inline function interaction_force_correction(system::WeaklyCompressibleSPHSystem,
                                              rho_a, rho_b)
    return free_surface_correction(system_correction(system), system, rho_a, rho_b)
end

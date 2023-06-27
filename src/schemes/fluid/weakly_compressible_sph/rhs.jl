# Fluid-fluid interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system, neighborhood_search,
                   particle_system::WeaklyCompressibleSPHSystem,
                   neighbor_system)
    @unpack density_calculator, state_equation, smoothing_length,
    correction, dv_pressure, dv_viscosity = particle_system
    @unpack sound_speed = state_equation

    viscosity = system_viscosity(neighbor_system)
    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    set_zero!(dv_pressure)
    set_zero!(dv_viscosity)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    for_particle_neighbor(particle_system, neighbor_system,
                          system_coords, neighbor_coords,
                          neighborhood_search) do particle, neighbor, pos_diff, distance
        # Only consider particles with a distance > 0.
        distance < sqrt(eps()) && return

        rho_a = particle_density(v_particle_system, particle_system, particle)
        rho_b = particle_density(v_neighbor_system, neighbor_system, neighbor)
        rho_mean = (rho_a + rho_b) / 2

        # Determine correction values
        viscosity_correction, pressure_correction = free_surface_correction(correction,
                                                                            particle_system,
                                                                            rho_mean)

        grad_kernel = smoothing_kernel_grad(particle_system, pos_diff, distance, particle)

        m_a = hydrodynamic_mass(particle_system, particle)
        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        @views dv_pressure[:, particle] += calc_pressure_eq(pressure_correction, m_b, particle,
                                                    particle_system, v_particle_system,
                                                    neighbor, neighbor_system,
                                                    v_neighbor_system, rho_a, rho_b,
                                                    pos_diff, distance, grad_kernel)

        @views dv_viscosity[:, particle] += viscosity_correction *
                                    viscosity(particle_system, neighbor_system,
                                              v_particle_system, v_neighbor_system,
                                              particle, neighbor, pos_diff, distance,
                                              sound_speed, m_a, m_b, rho_mean)

        continuity_equation!(dv, density_calculator,
                             v_particle_system, v_neighbor_system,
                             particle, neighbor, pos_diff, distance,
                             particle_system, neighbor_system, grad_kernel)
    end

    for particle in each_moving_particle(particle_system)
        for i in 1:ndims(particle_system)
            dv[i, particle] += dv_pressure[i, particle] + dv_viscosity[i, particle]
        end
    end

    return dv
end

@inline function calc_pressure_eq(pressure_correction, m_b, particle, particle_system,
                                  v_particle_system, neighbor,
                                  neighbor_system::WeaklyCompressibleSPHSystem,
                                  v_neighbor_system, rho_a, rho_b, pos_diff, distance,
                                  grad_kernel)
    return (-m_b *
            (particle_system.pressure[particle] / rho_a^2 +
             neighbor_system.pressure[neighbor] / rho_b^2) * grad_kernel) *
           pressure_correction
end

@inline function calc_pressure_eq(pressure_correction, m_b, particle, particle_system,
                                  v_particle_system, neighbor,
                                  neighbor_system::Union{BoundarySPHSystem,
                                                         TotalLagrangianSPHSystem},
                                  v_neighbor_system, rho_a, rho_b, pos_diff, distance,
                                  grad_kernel)
    @unpack boundary_model = neighbor_system

    return boundary_particle_impact(particle, neighbor, boundary_model,
                                    v_particle_system, v_neighbor_system,
                                    particle_system, neighbor_system,
                                    pos_diff, distance, m_b)
end

@inline function continuity_equation!(dv, density_calculator::ContinuityDensity,
                                      v_particle_system, v_neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      particle_system::WeaklyCompressibleSPHSystem,
                                      neighbor_system, grad_kernel)
    m_b = hydrodynamic_mass(neighbor_system, neighbor)
    vdiff = current_velocity(v_particle_system, particle_system, particle) -
            current_velocity(v_neighbor_system, neighbor_system, neighbor)
    dv[end, particle] += m_b * dot(vdiff, grad_kernel)

    return dv
end

@inline function continuity_equation!(dv, density_calculator::SummationDensity,
                                      v_particle_system, v_neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      particle_system, neighbor_system, grad_kernel)
    return dv
end

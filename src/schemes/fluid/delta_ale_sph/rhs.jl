# Discretization of equation (13) in Antuono et al. (2021).
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::DeltaALESPHSystem, neighbor_system, semi)
    (; correction) = particle_system

    sound_speed = system_sound_speed(particle_system)
    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)
    neighborhood_search = get_neighborhood_search(particle_system, neighbor_system, semi)
    backend = semi.parallelization_backend

    compact_support_ = compact_support(particle_system, neighbor_system)
    almostzero = sqrt(eps(compact_support_^2))

    @threaded semi for particle in each_integrated_particle(particle_system)
        m_a = @inbounds current_mass(v_particle_system, particle_system, particle)
        p_a = @inbounds current_pressure(v_particle_system, particle_system, particle)
        v_a = @inbounds current_velocity(v_particle_system, particle_system, particle)
        rho_a = @inbounds current_density(v_particle_system, particle_system, particle)

        dv_particle = Ref(zero(v_a))
        drho_particle = Ref(zero(rho_a))
        dm_particle = Ref(zero(m_a))

        @inbounds foreach_neighbor(system_coords, neighbor_coords, neighborhood_search,
                                   backend, particle
                                   ) do particle, neighbor, pos_diff,
                                        distance
            skip_zero_distance(particle_system) && distance < almostzero && return

            grad_kernel = smoothing_kernel_grad_unsafe(particle_system, pos_diff,
                                                       distance, particle)

            m_b = hydrodynamic_mass(neighbor_system, neighbor)
            v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)
            rho_b = current_density(v_neighbor_system, neighbor_system, neighbor)
            p_b = neighbor_pressure(v_neighbor_system, neighbor_system, neighbor, p_a)

            (viscosity_correction, pressure_correction,
             _) = free_surface_correction(correction, particle_system, rho_a, rho_b)

            dv_pressure = pressure_acceleration(particle_system, neighbor_system,
                                                particle, neighbor, m_a, m_b, p_a, p_b,
                                                rho_a, rho_b, pos_diff, distance,
                                                grad_kernel, correction)
            dv_particle[] += dv_pressure * pressure_correction

            dv_viscosity!(dv_particle, particle_system, neighbor_system,
                          v_particle_system, v_neighbor_system,
                          particle, neighbor, pos_diff, distance,
                          sound_speed, m_a, m_b, rho_a, rho_b,
                          v_a, v_b, grad_kernel, viscosity_correction)

            delta_ale_conservative_terms!(dv_particle, dm_particle,
                                          particle_system, neighbor_system,
                                          particle, neighbor, m_a, m_b, rho_a, rho_b,
                                          v_a, v_b, pos_diff, distance, grad_kernel)

            continuity_equation!(drho_particle, particle_system.density_calculator,
                                 particle_system, neighbor_system,
                                 particle, neighbor, pos_diff, distance,
                                 m_b, rho_a, rho_b, v_a, v_b, grad_kernel)
        end

        for i in eachindex(dv_particle[])
            @inbounds dv[i, particle] += dv_particle[][i]
        end
        @inbounds dv[ndims(particle_system) + 1, particle] += drho_particle[]
        @inbounds dv[ndims(particle_system) + 2, particle] += dm_particle[]
    end

    return dv
end

@propagate_inbounds function delta_ale_conservative_terms!(dv_particle, dm_particle,
                                                           particle_system,
                                                           neighbor_system,
                                                           particle, neighbor,
                                                           m_a, m_b, rho_a, rho_b,
                                                           v_a, v_b, pos_diff, distance,
                                                           grad_kernel)
    delta_v_a = delta_v(particle_system, particle)
    delta_v_b = delta_v(neighbor_system, neighbor)

    volume_a = div_fast(m_a, rho_a)
    volume_b = div_fast(m_b, rho_b)

    rho_delta_v = rho_a * delta_v_a + rho_b * delta_v_b
    dm_advection = volume_a * volume_b * dot(rho_delta_v, grad_kernel)
    dm_diffusion = mass_diffusion(particle_system, neighbor_system,
                                  particle, neighbor,
                                  m_a, m_b, rho_a, rho_b,
                                  pos_diff, distance, grad_kernel)
    dm = dm_advection + dm_diffusion

    momentum_advection = volume_a * volume_b *
                         (rho_a * v_a * dot(delta_v_a, grad_kernel) +
                          rho_b * v_b * dot(delta_v_b, grad_kernel))

    # Equation (13) evolves m*u. Convert it to the velocity derivative stored in `v`.
    dv_particle[] += div_fast(momentum_advection - v_a * dm, m_a)
    dm_particle[] += dm

    return dv_particle
end

@propagate_inbounds function mass_diffusion(particle_system::DeltaALESPHSystem,
                                            neighbor_system::DeltaALESPHSystem,
                                            particle, neighbor,
                                            m_a, m_b, rho_a, rho_b,
                                            pos_diff, distance, grad_kernel)
    particle_system === neighbor_system || return zero(m_a)

    volume_a = div_fast(m_a, rho_a)
    volume_b = div_fast(m_b, rho_b)
    h = (smoothing_length(particle_system, particle) +
         smoothing_length(neighbor_system, neighbor)) / 2
    coefficient = 2 * particle_system.density_diffusion.delta * h *
                  system_sound_speed(particle_system)

    # The paper uses r_ji = r_j - r_i, while `pos_diff` is r_i - r_j.
    F_ab = -div_fast(dot(pos_diff, grad_kernel), distance^2)

    return coefficient * (m_b - m_a) * F_ab * volume_a * volume_b
end

@inline function mass_diffusion(particle_system, neighbor_system,
                                particle, neighbor,
                                m_a, m_b, rho_a, rho_b,
                                pos_diff, distance, grad_kernel)
    return zero(m_a)
end

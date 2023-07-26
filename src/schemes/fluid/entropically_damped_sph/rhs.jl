# Fluid-fluid and fluid-boundary interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system, neighborhood_search,
                   particle_system::EntropicallyDampedSPHSystem,
                   neighbor_system)
    @unpack sound_speed = particle_system
    viscosity = viscosity_function(neighbor_system)

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    for_particle_neighbor(particle_system, neighbor_system,
                          system_coords, neighbor_coords,
                          neighborhood_search) do particle, neighbor, pos_diff, distance
        # Only consider particles with a distance > 0.
        distance < sqrt(eps()) && return

        rho_a = particle_density(v_particle_system, particle_system, particle)
        rho_b = particle_density(v_neighbor_system, neighbor_system, neighbor)

        p_a = particle_pressure(v_particle_system, particle_system, particle)
        p_b = particle_pressure(v_neighbor_system, neighbor_system, neighbor)

        m_a = hydrodynamic_mass(particle_system, particle)
        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        volume_a = m_a / rho_a
        volume_b = m_b / rho_b
        volume_term = (volume_a^2 + volume_b^2) / m_a

        # inter-particle averaged pressure
        p_avg = average_pressure(particle_system, particle)
        pressure_tilde = (rho_b * (p_a - p_avg) + rho_a * (p_b - p_avg)) / (rho_a + rho_b)

        grad_kernel = smoothing_kernel_grad(particle_system, pos_diff, distance)

        dv_pressure = -volume_term * pressure_tilde * grad_kernel

        dv_viscosity = viscosity(particle_system, neighbor_system,
                                 v_particle_system, v_neighbor_system,
                                 particle, neighbor, pos_diff, distance,
                                 sound_speed, m_a, m_b)

        dv_convection = momentum_convection(particle_system, neighbor_system,
                                            v_particle_system, v_neighbor_system, rho_a,
                                            rho_b, particle, neighbor, grad_kernel,
                                            volume_term)

        for i in 1:ndims(particle_system)
            dv[i, particle] += dv_pressure[i] + dv_viscosity[i] + dv_convection[i]
        end

        v_diff = current_velocity(v_particle_system, particle_system, particle) -
                 current_velocity(v_neighbor_system, neighbor_system, neighbor)

        pressure_evolution!(dv, particle_system, v_diff, grad_kernel,
                            particle, pos_diff, distance, sound_speed, volume_term, m_b,
                            p_a, p_b, rho_a, rho_b)

        transport_velocity!(dv, particle_system, volume_term, grad_kernel, particle)
    end

    return dv
end

@inline function pressure_evolution!(dv, particle_system, v_diff, grad_kernel, particle,
                                     pos_diff, distance, sound_speed, volume_term, m_b,
                                     p_a, p_b, rho_a, rho_b)
    @unpack smoothing_length = particle_system

    # EDAC pressure evolution
    pressure_diff = p_a - p_b

    artificial_eos = m_b * rho_a / rho_b * sound_speed^2 * dot(v_diff, grad_kernel)

    eta_a = rho_a * particle_system.nu_edac
    eta_b = rho_b * particle_system.nu_edac
    eta_tilde = 2 * eta_a * eta_b / (eta_a + eta_b)

    # TODO For variable smoothing length use average smoothing length
    tmp = eta_tilde / (distance^2 + 0.01 * smoothing_length^2)

    damping_term = volume_term * tmp * pressure_diff * dot(grad_kernel, pos_diff)

    dv[end, particle] += artificial_eos + damping_term

    return dv
end

@inline function transport_velocity!(dv, system::EntropicallyDampedSPHSystem, volume_term,
                                     grad_kernel, particle)
    transport_velocity!(dv, system, system.transport_velocity, volume_term, grad_kernel,
                        particle)
end

@inline transport_velocity!(dv, system, volume_term, grad_kernel, particle) = dv

@inline transport_velocity!(dv, system, ::Nothing, volume_term, grad_kernel, particle) = dv

@inline function transport_velocity!(dv, system, ::TransportVelocityAdami, volume_term,
                                     grad_kernel, particle)
    @unpack transport_velocity = system
    @unpack background_pressure = transport_velocity
    n_dims = ndims(system)

    for dim in 1:n_dims
        dv[n_dims + dim, particle] += dv[dim, particle]
        dv[n_dims + dim, particle] -= volume_term * background_pressure * grad_kernel[dim]
    end

    return dv
end

@inline function momentum_convection(system, neighbor_system,
                                     v_particle_system, v_neighbor_system, rho_a, rho_b,
                                     particle, neighbor, grad_kernel, volume_term)
    return SVector(ntuple(_ -> 0.0, Val(ndims(system))))
end

@inline function momentum_convection(system::EntropicallyDampedSPHSystem,
                                     neighbor_system::EntropicallyDampedSPHSystem,
                                     v_particle_system, v_neighbor_system, rho_a, rho_b,
                                     particle, neighbor, grad_kernel, volume_term)
    momentum_convection(system, neighbor_system, system.transport_velocity,
                        v_particle_system, v_neighbor_system, rho_a, rho_b,
                        particle, neighbor, grad_kernel, volume_term)
end

@inline function momentum_convection(system, neighbor_system, ::Nothing,
                                     v_particle_system, v_neighbor_system, rho_a, rho_b,
                                     particle, neighbor, grad_kernel, volume_term)
    return SVector(ntuple(_ -> 0.0, Val(ndims(system))))
end

@inline function momentum_convection(system, neighbor_system, ::TransportVelocityAdami,
                                     v_particle_system, v_neighbor_system, rho_a, rho_b,
                                     particle, neighbor, grad_kernel, volume_term)
    momentum_velocity_a = current_velocity(v_particle_system, system, particle)
    advection_velocity_a = advection_velocity(v_particle_system, system, particle)

    momentum_velocity_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)
    advection_velocity_b = advection_velocity(v_neighbor_system, neighbor_system, neighbor)

    A_a = rho_a * momentum_velocity_a * (advection_velocity_a - momentum_velocity_a)'
    A_b = rho_b * momentum_velocity_b * (advection_velocity_b - momentum_velocity_b)'

    return volume_term * (0.5 * (A_a + A_b)) * grad_kernel
end

@inline viscosity_function(system) = system.viscosity
@inline viscosity_function(system::BoundarySPHSystem) = system.boundary_model.viscosity

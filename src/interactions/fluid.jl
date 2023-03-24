function interact!(dv, v_particle_container, u_particle_container,
                   v_neighbor_container, u_neighbor_container, neighborhood_search,
                   particle_container::FluidParticleContainer, neighbor_container)
    @unpack density_calculator, smoothing_kernel, smoothing_length = particle_container

    @threaded for particle in each_moving_particle(particle_container)
        particle_coords = get_current_coords(particle, u_particle_container,
                                             particle_container)

        for neighbor in eachneighbor(particle_coords, neighborhood_search)
            neighbor_coords = get_current_coords(neighbor, u_neighbor_container,
                                                 neighbor_container)

            pos_diff = particle_coords - neighbor_coords
            distance2 = dot(pos_diff, pos_diff)

            if eps() < distance2 <= compact_support(smoothing_kernel, smoothing_length)^2
                distance = sqrt(distance2)

                calc_dv!(dv, v_particle_container, v_neighbor_container,
                         particle, neighbor, pos_diff, distance,
                         particle_container, neighbor_container)

                continuity_equation!(dv, density_calculator,
                                     v_particle_container, v_neighbor_container,
                                     particle, neighbor, pos_diff, distance,
                                     particle_container, neighbor_container)
            end
        end
    end

    return dv
end

# Fluid-fluid interaction
@inline function calc_dv!(dv, v_particle_container, v_neighbor_container,
                          particle, neighbor, pos_diff, distance,
                          particle_container::FluidParticleContainer,
                          neighbor_container::FluidParticleContainer)
    @unpack smoothing_kernel, smoothing_length,
    state_equation, viscosity = particle_container

    rho_a = get_particle_density(particle, v_particle_container,
                                 particle_container)
    rho_b = get_particle_density(neighbor, v_neighbor_container,
                                 neighbor_container)

    # Viscosity
    v_diff = get_particle_vel(particle, v_particle_container, particle_container) -
             get_particle_vel(neighbor, v_neighbor_container, neighbor_container)
    rho_mean = (rho_a + rho_b) / 2
    pi_ab = viscosity(state_equation.sound_speed, v_diff, pos_diff,
                      distance, rho_mean, smoothing_length)

    grad_kernel = kernel_grad(smoothing_kernel, pos_diff, distance, smoothing_length)
    m_b = neighbor_container.mass[neighbor]

    dv_pressure = -m_b *
                  (particle_container.pressure[particle] / rho_a^2 +
                   neighbor_container.pressure[neighbor] / rho_b^2) * grad_kernel
    dv_viscosity = -m_b * pi_ab * grad_kernel

    for i in 1:ndims(particle_container)
        dv[i, particle] += dv_pressure[i] + dv_viscosity[i]
    end

    return dv
end

# Fluid-boundary and fluid-solid interaction
@inline function calc_dv!(dv, v_particle_container, v_neighbor_container, particle,
                          neighbor, pos_diff, distance,
                          particle_container::FluidParticleContainer,
                          neighbor_container)
    @unpack smoothing_kernel, smoothing_length,
    state_equation = particle_container
    @unpack boundary_model = neighbor_container

    rho_a = get_particle_density(particle, v_particle_container,
                                 particle_container)

    # In fluid-solid interaction, use the "hydrodynamic mass" of the solid particles
    # corresponding to the rest density of the fluid and not the material density.
    m_b = get_hydrodynamic_mass(neighbor, neighbor_container)

    grad_kernel = kernel_grad(smoothing_kernel, pos_diff, distance, smoothing_length)

    dv_boundary = boundary_particle_impact(particle, neighbor,
                                           v_particle_container,
                                           v_neighbor_container,
                                           particle_container,
                                           neighbor_container,
                                           pos_diff, distance, m_b)

    dv_viscosity = calc_viscosity(boundary_model.viscosity,
                                  particle_container, neighbor_container,
                                  v_particle_container, v_neighbor_container,
                                  particle, neighbor, pos_diff, distance, rho_a,
                                  grad_kernel, state_equation.sound_speed, smoothing_length,
                                  m_b)

    for i in 1:ndims(particle_container)
        dv[i, particle] += dv_boundary[i] + dv_viscosity[i]
    end

    return dv
end

@inline function continuity_equation!(dv, density_calculator::ContinuityDensity,
                                      v_particle_container, v_neighbor_container,
                                      particle, neighbor, pos_diff, distance,
                                      particle_container::FluidParticleContainer,
                                      neighbor_container)
    @unpack smoothing_kernel, smoothing_length = particle_container

    mass = get_hydrodynamic_mass(neighbor, neighbor_container)
    vdiff = get_particle_vel(particle, v_particle_container, particle_container) -
            get_particle_vel(neighbor, v_neighbor_container, neighbor_container)
    NDIMS = ndims(particle_container)
    dv[NDIMS + 1, particle] += sum(mass * vdiff .*
                                   kernel_grad(smoothing_kernel, pos_diff, distance,
                                               smoothing_length))

    return dv
end

@inline function continuity_equation!(dv, density_calculator::SummationDensity,
                                      v_particle_container, v_neighbor_container,
                                      particle, neighbor, pos_diff, distance,
                                      particle_container, neighbor_container)
    return dv
end

@inline function calc_viscosity(viscosity::ArtificialViscosityMonaghan,
                                particle_container, neighbor_container,
                                v_particle_container, v_neighbor_container,
                                particle, neighbor, pos_diff, distance, rho_a,
                                grad_kernel, sound_speed, smoothing_length, m_b)
    v_diff = get_particle_vel(particle, v_particle_container, particle_container) -
             get_particle_vel(neighbor, v_neighbor_container, neighbor_container)

    pi_ab = viscosity(sound_speed, v_diff, pos_diff, distance, rho_a, smoothing_length)

    return -m_b * pi_ab * grad_kernel
end

@inline function calc_viscosity(viscosity::ViscousInteractionAdami,
                                particle_container, neighbor_container,
                                v_particle_container, v_neighbor_container,
                                particle, neighbor, pos_diff, distance, rho_a,
                                grad_kernel, sound_speed, smoothing_length, m_b)
    m_a = get_hydrodynamic_mass(particle, particle_container)
    rho_b = get_particle_density(neighbor, v_neighbor_container, neighbor_container)
    v_w = viscosity.velocities[:, neighbor]

    v_diff = get_particle_vel(particle, v_particle_container, particle_container) - v_w

    tmp =  viscosity.eta * v_diff / distance

    volume_a = m_a / rho_a
    volume_b = m_b / rho_b

    visc = (volume_a^2 + volume_b^2) * dot(grad_kernel, pos_diff / distance) .* tmp

    return visc/m_a
end

@inline function calc_viscosity(viscosity, particle_container, _, _, _, _, _, _, _, _, _, _,
                                _, _)
    return SVector(ntuple(_ -> 0.0, Val(ndims(particle_container))))
end

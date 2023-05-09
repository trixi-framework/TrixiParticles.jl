# Fluid-fluid interaction
function interact!(dv, v_particle_container, u_particle_container,
                   v_neighbor_container, u_neighbor_container, neighborhood_search,
                   particle_container::FluidParticleContainer,
                   neighbor_container::FluidParticleContainer)
    @unpack density_calculator, state_equation, viscosity,
    smoothing_length = particle_container

    container_coords = current_coordinates(u_particle_container, particle_container)
    neighbor_coords = current_coordinates(u_neighbor_container, neighbor_container)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    for_particle_neighbor(particle_container, neighbor_container,
                          container_coords, neighbor_coords,
                          neighborhood_search) do particle, neighbor, pos_diff, distance
        # Only consider particles with a distance > 0.
        distance < sqrt(eps()) && return

        rho_a = particle_density(v_particle_container, particle_container, particle)
        rho_b = particle_density(v_neighbor_container, neighbor_container, neighbor)

        # Viscosity
        v_diff = current_velocity(v_particle_container, particle_container, particle) -
                 current_velocity(v_neighbor_container, neighbor_container, neighbor)
        rho_mean = (rho_a + rho_b) / 2
        pi_ab = viscosity(state_equation.sound_speed, v_diff, pos_diff,
                          distance, rho_mean, smoothing_length)

        grad_kernel = smoothing_kernel_grad(particle_container, pos_diff, distance)
        m_b = neighbor_container.mass[neighbor]
        dv_pressure = -m_b *
                      (particle_container.pressure[particle] / rho_a^2 +
                       neighbor_container.pressure[neighbor] / rho_b^2) * grad_kernel
        dv_viscosity = -m_b * pi_ab * grad_kernel

        for i in 1:ndims(particle_container)
            dv[i, particle] += dv_pressure[i] + dv_viscosity[i]
        end

        continuity_equation!(dv, density_calculator,
                             v_particle_container, v_neighbor_container,
                             particle, neighbor, pos_diff, distance,
                             particle_container, neighbor_container)
    end

    return dv
end

@inline function continuity_equation!(dv, density_calculator::ContinuityDensity,
                                      v_particle_container, v_neighbor_container,
                                      particle, neighbor, pos_diff, distance,
                                      particle_container::FluidParticleContainer,
                                      neighbor_container)
    mass = hydrodynamic_mass(neighbor_container, neighbor)
    vdiff = current_velocity(v_particle_container, particle_container, particle) -
            current_velocity(v_neighbor_container, neighbor_container, neighbor)
    NDIMS = ndims(particle_container)
    dv[NDIMS + 1, particle] += sum(mass * vdiff .*
                                   smoothing_kernel_grad(particle_container, pos_diff,
                                                         distance))

    return dv
end

@inline function continuity_equation!(dv, density_calculator::SummationDensity,
                                      v_particle_container, v_neighbor_container,
                                      particle, neighbor, pos_diff, distance,
                                      particle_container, neighbor_container)
    return dv
end

# Fluid-boundary and fluid-solid interaction
function interact!(dv, v_particle_container, u_particle_container,
                   v_neighbor_container, u_neighbor_container, neighborhood_search,
                   particle_container::FluidParticleContainer,
                   neighbor_container::Union{BoundaryParticleContainer,
                                             SolidParticleContainer})
    @unpack density_calculator, state_equation, viscosity,
    smoothing_length = particle_container
    @unpack sound_speed = state_equation

    container_coords = current_coordinates(u_particle_container, particle_container)
    neighbor_coords = current_coordinates(u_neighbor_container, neighbor_container)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    for_particle_neighbor(particle_container, neighbor_container,
                          container_coords, neighbor_coords,
                          neighborhood_search) do particle, neighbor, pos_diff, distance
        # Only consider particles with a distance > 0.
        distance < sqrt(eps()) && return

        # In fluid-solid interaction, use the "hydrodynamic mass" of the solid particles
        # corresponding to the rest density of the fluid and not the material density.
        m_b = hydrodynamic_mass(neighbor_container, neighbor)

        v_a = current_velocity(v_particle_container, particle_container, particle)
        v_b = current_velocity(v_neighbor_container, neighbor_container, neighbor)
        v_diff = v_a - v_b

        rho_a = particle_density(v_particle_container, particle_container, particle)
        rho_b = particle_density(v_neighbor_container, neighbor_container, neighbor)
        rho_mean = (rho_a + rho_b) / 2
        pi_ab = viscosity(sound_speed, v_diff, pos_diff, distance, rho_mean,
                          smoothing_length)
        dv_viscosity = -m_b * pi_ab *
                       smoothing_kernel_grad(particle_container, pos_diff, distance)

        dv_boundary = boundary_particle_impact(particle, neighbor,
                                               v_particle_container, v_neighbor_container,
                                               particle_container, neighbor_container,
                                               pos_diff, distance, m_b)

        for i in 1:ndims(particle_container)
            dv[i, particle] += dv_boundary[i] + dv_viscosity[i]
        end

        continuity_equation!(dv, density_calculator,
                             v_particle_container, v_neighbor_container,
                             particle, neighbor, pos_diff, distance,
                             particle_container, neighbor_container)
    end

    return dv
end

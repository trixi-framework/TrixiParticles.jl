# Fluid-fluid interaction
function interact!(dv, v_particle_container, u_particle_container,
                   v_neighbor_container, u_neighbor_container, neighborhood_search,
                   particle_container::FluidParticleContainer,
                   neighbor_container::FluidParticleContainer)
    @unpack density_calculator, smoothing_kernel, smoothing_length = particle_container

    @threaded for particle in each_moving_particle(particle_container)
        particle_coords = current_coords(u_particle_container, particle_container,
                                         particle)
        for neighbor in eachneighbor(particle_coords, neighborhood_search)
            neighbor_coords = current_coords(u_neighbor_container, neighbor_container,
                                             neighbor)

            pos_diff = particle_coords - neighbor_coords
            distance = norm(pos_diff)

            if sqrt(eps()) < distance <= compact_support(smoothing_kernel, smoothing_length)
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

@inline function calc_dv!(dv, v_particle_container, v_neighbor_container,
                          particle, neighbor, pos_diff, distance,
                          particle_container, neighbor_container)
    @unpack smoothing_kernel, smoothing_length, state_equation, viscosity = particle_container

    density_particle = particle_density(v_particle_container, particle_container, particle)
    density_neighbor = particle_density(v_neighbor_container, neighbor_container, neighbor)

    # Viscosity
    v_diff = current_velocity(v_particle_container, particle_container, particle) -
             current_velocity(v_neighbor_container, neighbor_container, neighbor)
    density_mean = (density_particle + density_neighbor) / 2
    pi_ab = viscosity(state_equation.sound_speed, v_diff, pos_diff,
                      distance, density_mean, smoothing_length)

    grad_kernel = kernel_deriv(smoothing_kernel, distance, smoothing_length) * pos_diff /
                  distance
    m_b = neighbor_container.mass[neighbor]
    dv_pressure = -m_b *
                  (particle_container.pressure[particle] / density_particle^2 +
                   neighbor_container.pressure[neighbor] / density_neighbor^2) * grad_kernel
    dv_viscosity = -m_b * pi_ab * grad_kernel

    for i in 1:ndims(particle_container)
        dv[i, particle] += dv_pressure[i] + dv_viscosity[i]
    end

    return dv
end

@inline function continuity_equation!(dv, density_calculator::ContinuityDensity,
                                      v_particle_container, v_neighbor_container,
                                      particle, neighbor, pos_diff, distance,
                                      particle_container::FluidParticleContainer,
                                      neighbor_container)
    @unpack smoothing_kernel, smoothing_length = particle_container

    mass = hydrodynamic_mass(neighbor_container, neighbor)
    vdiff = current_velocity(v_particle_container, particle_container, particle) -
            current_velocity(v_neighbor_container, neighbor_container, neighbor)
    NDIMS = ndims(particle_container)
    dv[NDIMS + 1, particle] += sum(mass * vdiff *
                                   kernel_deriv(smoothing_kernel,
                                                distance,
                                                smoothing_length) .*
                                   pos_diff) / distance

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
    smoothing_kernel, smoothing_length = particle_container
    @unpack sound_speed = state_equation

    @threaded for particle in each_moving_particle(particle_container)
        density_a = particle_density(v_particle_container, particle_container, particle)
        v_a = current_velocity(v_particle_container, particle_container, particle)

        particle_coords = current_coords(u_particle_container, particle_container,
                                         particle)
        for neighbor in eachneighbor(particle_coords, neighborhood_search)
            neighbor_coords = current_coords(u_neighbor_container, neighbor_container,
                                             neighbor)

            pos_diff = particle_coords - neighbor_coords
            distance = norm(pos_diff)

            if sqrt(eps()) < distance <= compact_support(smoothing_kernel, smoothing_length)

                # In fluid-solid interaction, use the "hydrodynamic mass" of the solid particles
                # corresponding to the rest density of the fluid and not the material density.
                m_b = hydrodynamic_mass(neighbor_container, neighbor)
                v_b = current_velocity(v_neighbor_container, neighbor_container, neighbor)
                v_diff = v_a - v_b

                continuity_equation!(dv, density_calculator,
                                     v_particle_container, v_neighbor_container,
                                     particle, neighbor, pos_diff, distance,
                                     particle_container, neighbor_container)

                pi_ab = viscosity(sound_speed, v_diff, pos_diff, distance, density_a,
                                  smoothing_length)
                dv_viscosity = -m_b * pi_ab *
                               kernel_deriv(smoothing_kernel, distance, smoothing_length) *
                               pos_diff / distance

                dv_boundary = boundary_particle_impact(particle, neighbor,
                                                       v_particle_container,
                                                       v_neighbor_container,
                                                       particle_container,
                                                       neighbor_container,
                                                       pos_diff, distance, m_b)

                for i in 1:ndims(particle_container)
                    dv[i, particle] += dv_boundary[i] + dv_viscosity[i]
                end
            end
        end
    end

    return dv
end

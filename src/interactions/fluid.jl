function interact!(dv, v_particle_container, u_particle_container,
                   v_neighbor_container, u_neighbor_container, neighborhood_search,
                   particle_container::FluidParticleContainer,
                   neighbor_container::FluidParticleContainer)
    @unpack density_calculator = particle_container

    @threaded for particle in each_moving_particle(particle_container)
        particle_coords = current_coords(u_particle_container, particle_container,
                                         particle)
        for neighbor in eachneighbor(particle_coords, neighborhood_search)
            neighbor_coords = current_coords(u_neighbor_container, neighbor_container,
                                             neighbor)

            pos_diff = particle_coords - neighbor_coords
            distance2 = dot(pos_diff, pos_diff)

            if eps() < distance2 <= compact_support(particle_container)^2
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

@inline function calc_dv!(dv, v_particle_container, v_neighbor_container,
                          particle, neighbor, pos_diff, distance,
                          particle_container, neighbor_container)
    @unpack state_equation, viscosity, smoothing_length = particle_container

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

    @threaded for particle in each_moving_particle(particle_container)
        rho_a = particle_density(v_particle_container, particle_container, particle)
        v_a = current_velocity(v_particle_container, particle_container, particle)

        particle_coords = current_coords(u_particle_container, particle_container,
                                         particle)
        for neighbor in eachneighbor(particle_coords, neighborhood_search)
            neighbor_coords = current_coords(u_neighbor_container, neighbor_container,
                                             neighbor)

            pos_diff = particle_coords - neighbor_coords
            distance2 = dot(pos_diff, pos_diff)

            if eps() < distance2 <= compact_support(particle_container)^2
                distance = sqrt(distance2)

                # In fluid-solid interaction, use the "hydrodynamic mass" of the solid particles
                # corresponding to the rest density of the fluid and not the material density.
                m_b = hydrodynamic_mass(neighbor_container, neighbor)
                rho_b = particle_density(v_neighbor_container,
                                         neighbor_container, neighbor)
                v_b = current_velocity(v_neighbor_container, neighbor_container, neighbor)
                v_diff = v_a - v_b

                rho_mean = (rho_a + rho_b) / 2
                pi_ab = viscosity(state_equation.sound_speed, v_diff, pos_diff,
                                  distance, rho_mean, smoothing_length)
                dv_viscosity = -m_b * pi_ab *
                               smoothing_kernel_grad(particle_container, pos_diff, distance)

                calc_dv!(dv, neighbor, neighbor_container, particle, particle_container,
                         rho_a, rho_b, m_b, grad_kernel, pi_ab, v_particle_container,
                         v_neighbor_container, pos_diff, distance)

                for i in 1:ndims(particle_container)
                    dv[i, particle] += dv_boundary[i] + dv_viscosity[i]
                end

                continuity_equation!(dv, density_calculator,
                                     v_particle_container, v_neighbor_container,
                                     particle, neighbor, pos_diff, distance,
                                     particle_container, neighbor_container)
            end
        end
    end

    return dv
end

@inline function calc_dv!(dv, neighbor, neighbor_container::FluidParticleContainer,
                          particle, particle_container, rho_a, rho_b, m_b, grad_kernel,
                          pi_ab, v_particle_container, v_neighbor_container, pos_diff,
                          distance)
    dv_viscosity = -m_b * pi_ab * grad_kernel

    p_a = particle_container.pressure[particle]
    p_b = neighbor_container.pressure[neighbor]
    dv_pressure = -m_b * (p_a / rho_a^2 + p_b / rho_b^2) * grad_kernel

    for i in 1:ndims(particle_container)
        dv[i, particle] += dv_pressure[i] + dv_viscosity[i]
    end
end

@inline function calc_dv!(dv, neighbor,
                          neighbor_container::Union{BoundaryParticleContainer,
                                                    SolidParticleContainer}, particle,
                          particle_container, rho_a, rho_b, m_b, grad_kernel, pi_ab,
                          v_particle_container, v_neighbor_container, pos_diff, distance)
    dv_viscosity = -m_b * pi_ab * grad_kernel

    dv_boundary = boundary_particle_impact(particle, neighbor, v_particle_container,
                                           v_neighbor_container, particle_container,
                                           neighbor_container, pos_diff, distance, m_b)

    for i in 1:ndims(particle_container)
        dv[i, particle] += dv_boundary[i] + dv_viscosity[i]
    end
end

@inline function continuity_equation!(dv, density_calculator::ContinuityDensity,
                                      particle, neighbor, v_diff, m_b,
                                      particle_container::FluidParticleContainer,
                                      neighbor_container, grad_kernel)
    NDIMS = ndims(particle_container)
    dv[NDIMS + 1, particle] += sum(m_b * vdiff .* grad_kernel)

    return dv
end

@inline function continuity_equation!(dv, density_calculator::SummationDensity,
                                      particle, neighbor, v_diff, m_b, particle_container,
                                      neighbor_container, grad_kernel)
    return dv
end

# Fluid-fluid interaction
function interact!(dv, v_particle_container, u_particle_container,
                   v_neighbor_container, u_neighbor_container, neighborhood_search,
                   particle_container::FluidParticleContainer,
                   neighbor_container::FluidParticleContainer)
    @unpack density_calculator, smoothing_kernel, smoothing_length = particle_container

    @threaded for particle in each_moving_particle(particle_container)
        particle_coords = get_current_coords(particle, u_particle_container,
                                             particle_container)
        for neighbor in eachneighbor(particle_coords, neighborhood_search)
            neighbor_coords = get_current_coords(neighbor, u_neighbor_container,
                                                 neighbor_container)

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

    density_particle = get_particle_density(particle, v_particle_container,
                                            particle_container)
    density_neighbor = get_particle_density(neighbor, v_neighbor_container,
                                            neighbor_container)

    # Viscosity
    v_diff = get_particle_vel(particle, v_particle_container, particle_container) -
             get_particle_vel(neighbor, v_neighbor_container, neighbor_container)
    density_mean = (density_particle + density_neighbor) / 2
    pi_ab = viscosity(state_equation.sound_speed, v_diff, pos_diff,
                      distance, density_mean, smoothing_length)

    grad_kernel = kernel_grad(smoothing_kernel, pos_diff, distance, smoothing_length)
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
        density_a = get_particle_density(particle, v_particle_container, particle_container)
        v_a = get_particle_vel(particle, v_particle_container, particle_container)

        particle_coords = get_current_coords(particle, u_particle_container,
                                             particle_container)
        for neighbor in eachneighbor(particle_coords, neighborhood_search)
            neighbor_coords = get_current_coords(neighbor, u_neighbor_container,
                                                 neighbor_container)

            pos_diff = particle_coords - neighbor_coords
            distance = norm(pos_diff)

            if sqrt(eps()) < distance <= compact_support(smoothing_kernel, smoothing_length)

                # In fluid-solid interaction, use the "hydrodynamic mass" of the solid particles
                # corresponding to the rest density of the fluid and not the material density.
                m_b = get_hydrodynamic_mass(neighbor, neighbor_container)
                v_b = get_particle_vel(neighbor, v_neighbor_container, neighbor_container)
                v_diff = v_a - v_b

                continuity_equation!(dv, density_calculator,
                                     v_particle_container, v_neighbor_container,
                                     particle, neighbor, pos_diff, distance,
                                     particle_container, neighbor_container)

                pi_ab = viscosity(sound_speed, v_diff, pos_diff, distance, density_a,
                                  smoothing_length)
                dv_viscosity = -m_b * pi_ab *
                               kernel_grad(smoothing_kernel, pos_diff, distance,
                                           smoothing_length)

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

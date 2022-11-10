# Fluid-fluid interaction
function calc_du!(du, u_particle_container, u_neighbor_container,
                  particle_container::FluidParticleContainer,
                  neighbor_container::FluidParticleContainer)
    @unpack smoothing_kernel, smoothing_length = particle_container
    @unpack neighborhood_search = neighbor_container

    @threaded for particle in each_moving_particle(particle_container)
        particle_coords = get_current_coords(particle, u_particle_container, particle_container)
        for neighbor in eachneighbor(particle_coords, neighborhood_search)
            neighbor_coords = get_current_coords(neighbor, u_neighbor_container, neighbor_container)

            pos_diff = particle_coords - neighbor_coords
            distance = norm(pos_diff)

            if sqrt(eps()) < distance <= compact_support(smoothing_kernel, smoothing_length)
                calc_dv!(du, u_particle_container, u_neighbor_container,
                         particle, neighbor, pos_diff, distance,
                         particle_container, neighbor_container)

                # TODO
                # continuity_equation!(du, u, particle, neighbor, pos_diff, distance, semi)
            end
        end
    end
end


@inline function calc_dv!(du, u_particle_container, u_neighbor_container,
                          particle, neighbor, pos_diff, distance,
                          particle_container, neighbor_container)
    @unpack smoothing_kernel, smoothing_length, state_equation, viscosity = particle_container

    density_particle = get_particle_density(particle, u_particle_container, particle_container)
    density_neighbor = get_particle_density(neighbor, u_neighbor_container, neighbor_container)

    # Viscosity
    v_diff = get_particle_vel(particle, u_particle_container, particle_container) -
        get_particle_vel(neighbor, u_neighbor_container, neighbor_container)
    density_mean = (density_particle + density_neighbor) / 2
    pi_ab = viscosity(state_equation.sound_speed, v_diff, pos_diff,
                      distance, density_mean, smoothing_length)

    grad_kernel = kernel_deriv(smoothing_kernel, distance, smoothing_length) * pos_diff / distance
    m_b = neighbor_container.mass[neighbor]
    dv_pressure = -m_b * (particle_container.pressure[particle] / density_particle^2 +
                          neighbor_container.pressure[neighbor] / density_neighbor^2) * grad_kernel
    dv_viscosity = m_b * pi_ab * grad_kernel

    dv = dv_pressure + dv_viscosity

    for i in 1:ndims(particle_container)
        du[ndims(particle_container) + i, particle] += dv[i]
    end

    return du
end


# TODO
@inline function continuity_equation!(du, u_particle_container, u_neighbor_container,
                                      particle, neighbor, pos_diff, distance,
                                      semi)
    @unpack smoothing_kernel, smoothing_length, cache = semi
    @unpack mass = cache

    vdiff = get_particle_vel(u, semi, particle) -
            get_particle_vel(u, semi, neighbor)

    du[2 * ndims(semi) + 1, particle] += sum(mass[particle] * vdiff *
                                             kernel_deriv(smoothing_kernel, distance, smoothing_length) .*
                                             pos_diff) / distance

    return du
end

@inline function continuity_equation!(du, u, particle, neighbor, pos_diff, distance,
                                      semi)
    return du
end


# Fluid-boundary interaction
function calc_du!(du, u_particle_container, u_neighbor_container,
                  particle_container::FluidParticleContainer,
                  neighbor_container::BoundaryParticleContainer)
    @unpack state_equation, viscosity, smoothing_kernel, smoothing_length = particle_container
    @unpack neighborhood_search = neighbor_container

    @threaded for particle in each_moving_particle(particle_container)

        m_a = particle_container.mass[particle]
        density_a = get_particle_density(particle, u_particle_container, particle_container)
        v_a = get_particle_vel(particle, u_particle_container, particle_container)

        particle_coords = get_current_coords(particle, u_particle_container, particle_container)
        for neighbor in eachneighbor(particle_coords, neighborhood_search)
            neighbor_coords = get_current_coords(neighbor, u_neighbor_container, neighbor_container)

            pos_diff = particle_coords - neighbor_coords
            distance = norm(pos_diff)

            if sqrt(eps()) < distance <= compact_support(smoothing_kernel, smoothing_length)
                m_b = neighbor_container.mass[neighbor]

                dv = boundary_particle_impact(particle, particle_container, neighbor_container, pos_diff, distance,
                                              m_a, m_b, density_a, v_a)

                for i in 1:ndims(particle_container)
                    du[ndims(particle_container) + i, particle] += dv[i]
                end
            end
        end
    end
end

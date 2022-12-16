# Solid-solid interaction
function interact!(du, u_particle_container, u_neighbor_container, neighborhood_search,
                   particle_container::SolidParticleContainer,
                   neighbor_container::SolidParticleContainer)
    @unpack smoothing_kernel, smoothing_length, penalty_force = particle_container

    # Different solids do not interact with each other (yet)
    if particle_container !== neighbor_container
        return du
    end

    @threaded for particle in each_moving_particle(particle_container)
        # Everything here is done in the initial coordinates
        particle_coords = get_particle_coords(particle, particle_container.initial_coordinates,
                                              particle_container)
        for neighbor in eachneighbor(particle_coords, neighborhood_search)
            neighbor_coords = get_particle_coords(neighbor, neighbor_container.initial_coordinates,
                                                  neighbor_container)

            pos_diff = particle_coords - neighbor_coords
            distance = norm(pos_diff)

            if sqrt(eps()) < distance <= compact_support(smoothing_kernel, smoothing_length)
                calc_dv!(du, particle, neighbor, pos_diff, distance,
                         particle_container, neighbor_container)
                calc_penalty_force!(du, particle, neighbor, pos_diff,
                                    distance, particle_container, penalty_force)
            end
        end
    end

    return du
end

@inline function calc_dv!(du, particle, neighbor, initial_pos_diff, initial_distance,
                          particle_container, neighbor_container)
    @unpack smoothing_kernel, smoothing_length = particle_container

    density_particle = particle_container.material_density[particle]
    density_neighbor = neighbor_container.material_density[neighbor]

    grad_kernel = kernel_deriv(smoothing_kernel, initial_distance, smoothing_length) *
        initial_pos_diff / initial_distance

    m_b = neighbor_container.mass[neighbor]

    dv = m_b * (get_pk1_corrected(particle, particle_container) / density_particle^2 +
                get_pk1_corrected(neighbor, neighbor_container) / density_neighbor^2) * grad_kernel

    for i in 1:ndims(particle_container)
        du[ndims(particle_container) + i, particle] += dv[i]
    end

    return du
end


# Solid-fluid interaction
function interact!(du, u_particle_container, u_neighbor_container, neighborhood_search,
                   particle_container::SolidParticleContainer,
                   neighbor_container::FluidParticleContainer)
    @unpack density_calculator, state_equation, viscosity, smoothing_kernel, smoothing_length = neighbor_container

    @threaded for particle in each_moving_particle(particle_container)
        particle_coords = get_current_coords(particle, u_particle_container, particle_container)
        for neighbor in eachneighbor(particle_coords, neighborhood_search)
            m_b = neighbor_container.mass[neighbor]
            density_b = get_particle_density(neighbor, u_neighbor_container, neighbor_container)

            neighbor_coords = get_current_coords(neighbor, u_neighbor_container, neighbor_container)

            pos_diff = particle_coords - neighbor_coords
            distance = norm(pos_diff)

            if sqrt(eps()) < distance <= compact_support(smoothing_kernel, smoothing_length)
                # Apply the same force to the solid particle
                # that the fluid particle experiences due to the soild particle.
                # Note that the same arguments are passed here as in fluid-solid interact!,
                # except that m_b is now the fluid mass and pos_diff has a flipped sign.
                dv = boundary_particle_impact(neighbor, neighbor_container, particle_container,
                                              pos_diff, distance, density_b, m_b)

                for i in 1:ndims(particle_container)
                    du[ndims(particle_container) + i, particle] += dv[i]
                end

                # TODO
                # continuity_equation!(du, density_calculator,
                #                      u_particle_container, u_particle_container,
                #                      particle, neighbor, pos_diff, distance,
                #                      neighbor_container, particle_container)
            end
        end
    end

    return du
end


# Solid-boundary interaction
function interact!(du, u_particle_container, u_neighbor_container, neighborhood_search,
                   particle_container::SolidParticleContainer,
                   neighbor_container::BoundaryParticleContainer)
    return du
end

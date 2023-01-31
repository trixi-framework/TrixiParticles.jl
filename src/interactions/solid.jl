# Solid-solid interaction
function interact!(du, u_particle_container, u_neighbor_container, neighborhood_search,
                   particle_container::SolidParticleContainer,
                   neighbor_container::SolidParticleContainer)
    interact_solid_solid!(du, neighborhood_search, particle_container, neighbor_container)
end

# Function barrier without dispatch for unit testing
@inline function interact_solid_solid!(du, neighborhood_search, particle_container,
                                       neighbor_container)
    @unpack smoothing_kernel, smoothing_length, penalty_force = particle_container

    # Different solids do not interact with each other (yet)
    if particle_container !== neighbor_container
        return du
    end

    @threaded for particle in each_moving_particle(particle_container)
        # Everything here is done in the initial coordinates
        particle_coords = get_particle_coords(particle,
                                              particle_container.initial_coordinates,
                                              particle_container)
        for neighbor in eachneighbor(particle_coords, neighborhood_search)
            neighbor_coords = get_particle_coords(neighbor,
                                                  neighbor_container.initial_coordinates,
                                                  neighbor_container)

            pos_diff = particle_coords - neighbor_coords
            distance = norm(pos_diff)

            if sqrt(eps()) < distance <= compact_support(smoothing_kernel, smoothing_length)
                calc_dv!(du, particle, neighbor, pos_diff, distance,
                         particle_container, neighbor_container)

                calc_penalty_force!(du, particle, neighbor, pos_diff,
                                    distance, particle_container, penalty_force)

                # TODO continuity equation?
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

    dv = m_b *
         (get_pk1_corrected(particle, particle_container) / density_particle^2 +
          get_pk1_corrected(neighbor, neighbor_container) / density_neighbor^2) *
         grad_kernel

    for i in 1:ndims(particle_container)
        du[ndims(particle_container) + i, particle] += dv[i]
    end

    return du
end

# Solid-fluid interaction
function interact!(du, u_particle_container, u_neighbor_container, neighborhood_search,
                   particle_container::SolidParticleContainer,
                   neighbor_container::FluidParticleContainer)
    @unpack state_equation, viscosity, smoothing_kernel, smoothing_length = neighbor_container
    @unpack boundary_model = particle_container

    @threaded for particle in each_moving_particle(particle_container)
        m_a = get_hydrodynamic_mass(particle, particle_container)
        v_a = get_particle_vel(particle, u_particle_container, particle_container)

        particle_coords = get_current_coords(particle, u_particle_container,
                                             particle_container)
        for neighbor in eachneighbor(particle_coords, neighborhood_search)
            neighbor_coords = get_current_coords(neighbor, u_neighbor_container,
                                                 neighbor_container)

            pos_diff = particle_coords - neighbor_coords
            distance = norm(pos_diff)

            if sqrt(eps()) < distance <= compact_support(smoothing_kernel, smoothing_length)

                # Apply the same force to the solid particle
                # that the fluid particle experiences due to the soild particle.
                # Note that the same arguments are passed here as in fluid-solid interact!,
                # except that pos_diff has a flipped sign.
                m_b = get_hydrodynamic_mass(neighbor, neighbor_container)
                density_b = get_particle_density(neighbor, u_neighbor_container,
                                                 neighbor_container)
                v_b = get_particle_vel(neighbor, u_neighbor_container, neighbor_container)
                v_diff = v_b - v_a

                pi_ab = viscosity(state_equation.sound_speed, v_diff, pos_diff, distance,
                                  density_b, smoothing_length)
                dv_viscosity = -m_a * pi_ab *
                               kernel_deriv(smoothing_kernel, distance, smoothing_length) *
                               pos_diff / distance
                dv_boundary = boundary_particle_impact(neighbor, particle,
                                                       u_neighbor_container,
                                                       u_particle_container,
                                                       neighbor_container,
                                                       particle_container,
                                                       pos_diff, distance, m_a)
                dv = dv_boundary + dv_viscosity

                for i in 1:ndims(particle_container)
                    # Multiply dv (acceleration on fluid particle b) by m_b to obtain the force
                    # Divide by m_a to obtain the acceleration of solid particle a
                    du[ndims(particle_container) + i, particle] += dv[i] * m_b /
                                                                   particle_container.mass[particle]
                end

                continuity_equation!(du, boundary_model,
                                     u_particle_container, u_neighbor_container,
                                     particle, neighbor, pos_diff, distance,
                                     particle_container, neighbor_container)
            end
        end
    end

    return du
end

@inline function continuity_equation!(du, boundary_model,
                                      u_particle_container, u_neighbor_container,
                                      particle, neighbor, pos_diff, distance,
                                      particle_container::SolidParticleContainer,
                                      neighbor_container::FluidParticleContainer)
    return du
end

@inline function continuity_equation!(du, boundary_model::BoundaryModelDummyParticles,
                                      u_particle_container, u_neighbor_container,
                                      particle, neighbor, pos_diff, distance,
                                      particle_container::SolidParticleContainer,
                                      neighbor_container::FluidParticleContainer)
    @unpack density_calculator = boundary_model

    continuity_equation!(du, density_calculator,
                         u_particle_container, u_neighbor_container,
                         particle, neighbor, pos_diff, distance,
                         particle_container, neighbor_container)
end

@inline function continuity_equation!(du, ::ContinuityDensity,
                                      u_particle_container, u_neighbor_container,
                                      particle, neighbor, pos_diff, distance,
                                      particle_container::SolidParticleContainer,
                                      neighbor_container::FluidParticleContainer)
    @unpack smoothing_kernel, smoothing_length = neighbor_container

    vdiff = get_particle_vel(particle, u_particle_container, particle_container) -
            get_particle_vel(neighbor, u_neighbor_container, neighbor_container)

    du[2 * ndims(particle_container) + 1, particle] += sum(neighbor_container.mass[neighbor] *
                                                           vdiff *
                                                           kernel_deriv(smoothing_kernel,
                                                                        distance,
                                                                        smoothing_length) .*
                                                           pos_diff) / distance

    return du
end

@inline function continuity_equation!(du, ::SummationDensity,
                                      u_particle_container, u_neighbor_container,
                                      particle, neighbor, pos_diff, distance,
                                      particle_container::SolidParticleContainer,
                                      neighbor_container::FluidParticleContainer)
    return du
end

# Solid-boundary interaction
function interact!(du, u_particle_container, u_neighbor_container, neighborhood_search,
                   particle_container::SolidParticleContainer,
                   neighbor_container::BoundaryParticleContainer)
    # TODO continuity equation?
    return du
end

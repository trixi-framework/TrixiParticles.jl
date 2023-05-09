# Solid-solid interaction
function interact!(dv, v_particle_container, u_particle_container,
                   v_neighbor_container, u_neighbor_container, neighborhood_search,
                   particle_container::SolidParticleContainer,
                   neighbor_container::SolidParticleContainer)
    interact_solid_solid!(dv, neighborhood_search, particle_container, neighbor_container)
end

# Function barrier without dispatch for unit testing
@inline function interact_solid_solid!(dv, neighborhood_search, particle_container,
                                       neighbor_container)
    @unpack penalty_force = particle_container

    # Different solids do not interact with each other (yet)
    if particle_container !== neighbor_container
        return dv
    end

    @threaded for particle in each_moving_particle(particle_container)
        # Everything here is done in the initial coordinates
        particle_coords = initial_coords(particle_container, particle)
        for neighbor in eachneighbor(particle_coords, neighborhood_search)
            neighbor_coords = initial_coords(neighbor_container, neighbor)

            pos_diff = particle_coords - neighbor_coords
            distance2 = dot(pos_diff, pos_diff)

            if eps() < distance2 <= compact_support(particle_container)^2
                distance = sqrt(distance2)

                calc_dv!(dv, particle, neighbor, pos_diff, distance,
                         particle_container, neighbor_container)

                calc_penalty_force!(dv, particle, neighbor, pos_diff,
                                    distance, particle_container, penalty_force)

                # TODO continuity equation?
            end
        end
    end

    return dv
end

@inline function calc_dv!(dv, particle, neighbor, initial_pos_diff, initial_distance,
                          particle_container, neighbor_container)
    rho_a = particle_container.material_density[particle]
    rho_b = neighbor_container.material_density[neighbor]

    grad_kernel = smoothing_kernel_grad(particle_container, initial_pos_diff,
                                        initial_distance)

    m_b = neighbor_container.mass[neighbor]

    dv_particle = m_b *
                  (pk1_corrected(particle_container, particle) / rho_a^2 +
                   pk1_corrected(neighbor_container, neighbor) / rho_b^2) *
                  grad_kernel

    for i in 1:ndims(particle_container)
        dv[i, particle] += dv_particle[i]
    end

    return dv
end

# Solid-fluid interaction
function interact!(dv, v_particle_container, u_particle_container,
                   v_neighbor_container, u_neighbor_container, neighborhood_search,
                   particle_container::SolidParticleContainer,
                   neighbor_container::FluidParticleContainer)
    @unpack state_equation, viscosity, smoothing_length = neighbor_container
    @unpack boundary_model = particle_container

    @threaded for particle in each_moving_particle(particle_container)
        m_a = hydrodynamic_mass(particle_container, particle)
        v_a = current_velocity(v_particle_container, particle_container, particle)

        particle_coords = current_coords(u_particle_container, particle_container,
                                         particle)
        for neighbor in eachneighbor(particle_coords, neighborhood_search)
            neighbor_coords = current_coords(u_neighbor_container, neighbor_container,
                                             neighbor)

            pos_diff = particle_coords - neighbor_coords
            distance2 = dot(pos_diff, pos_diff)

            if eps() < distance2 <= compact_support(neighbor_container)^2
                distance = sqrt(distance2)

                # Apply the same force to the solid particle
                # that the fluid particle experiences due to the soild particle.
                # Note that the same arguments are passed here as in fluid-solid interact!,
                # except that pos_diff has a flipped sign.
                rho_b = particle_density(v_neighbor_container, neighbor_container,
                                         neighbor)
                v_b = current_velocity(v_neighbor_container, neighbor_container, neighbor)

                # Flip sign to get the same force as for the fluid-solid direction.
                v_diff = -(v_a - v_b)

                pi_ab = viscosity(state_equation.sound_speed, v_diff, pos_diff, distance,
                                  rho_b, smoothing_length)

                # use `m_a` to get the same viscosity as for the fluid-solid direction.
                dv_viscosity = -m_a * pi_ab *
                               smoothing_kernel_grad(neighbor_container, pos_diff, distance)
                dv_boundary = boundary_particle_impact(neighbor, particle,
                                                       v_neighbor_container,
                                                       v_particle_container,
                                                       neighbor_container,
                                                       particle_container,
                                                       pos_diff, distance, m_a)
                dv_particle = dv_boundary + dv_viscosity

                for i in 1:ndims(particle_container)
                    # Multiply `dv` (acceleration on fluid particle b) by the mass of
                    # particle b to obtain the force.
                    # Divide by the material mass of particle a to obtain the acceleration
                    # of solid particle a.
                    dv[i, particle] += dv_particle[i] * neighbor_container.mass[neighbor] /
                                       particle_container.mass[particle]
                end

                continuity_equation!(dv, boundary_model,
                                     v_particle_container, v_neighbor_container,
                                     particle, neighbor, pos_diff, distance,
                                     particle_container, neighbor_container)
            end
        end
    end

    return dv
end

@inline function continuity_equation!(dv, boundary_model,
                                      v_particle_container, v_neighbor_container,
                                      particle, neighbor, pos_diff, distance,
                                      particle_container::SolidParticleContainer,
                                      neighbor_container::FluidParticleContainer)
    return dv
end

@inline function continuity_equation!(dv, boundary_model::BoundaryModelDummyParticles,
                                      v_particle_container, v_neighbor_container,
                                      particle, neighbor, pos_diff, distance,
                                      particle_container::SolidParticleContainer,
                                      neighbor_container::FluidParticleContainer)
    @unpack density_calculator = boundary_model

    continuity_equation!(dv, density_calculator,
                         v_particle_container, v_neighbor_container,
                         particle, neighbor, pos_diff, distance,
                         particle_container, neighbor_container)
end

@inline function continuity_equation!(dv, ::ContinuityDensity,
                                      v_particle_container, v_neighbor_container,
                                      particle, neighbor, pos_diff, distance,
                                      particle_container::SolidParticleContainer,
                                      neighbor_container::FluidParticleContainer)
    vdiff = current_velocity(v_particle_container, particle_container, particle) -
            current_velocity(v_neighbor_container, neighbor_container, neighbor)

    NDIMS = ndims(particle_container)
    dv[NDIMS + 1, particle] += sum(neighbor_container.mass[neighbor] * vdiff .*
                                   smoothing_kernel_grad(neighbor_container, pos_diff,
                                                         distance))

    return dv
end

@inline function continuity_equation!(du, ::SummationDensity,
                                      u_particle_container, u_neighbor_container,
                                      particle, neighbor, pos_diff, distance,
                                      particle_container::SolidParticleContainer,
                                      neighbor_container::FluidParticleContainer)
    return du
end

# Solid-boundary interaction
function interact!(dv, v_particle_container, u_particle_container,
                   v_neighbor_container, u_neighbor_container, neighborhood_search,
                   particle_container::SolidParticleContainer,
                   neighbor_container::BoundaryParticleContainer)
    # TODO continuity equation?
    return dv
end

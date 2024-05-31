
# Solid-fluid interaction
# Since the object is rigid the movement gets averaged across all particles.
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system, neighborhood_search,
                   particle_system::RigidSPHSystem,
                   neighbor_system::FluidSystem)
    sound_speed = system_sound_speed(neighbor_system)

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    total_dv = zeros(ndims(particle_system))

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    for_particle_neighbor(particle_system, neighbor_system,
                          system_coords, neighbor_coords,
                          neighborhood_search) do particle, neighbor, pos_diff, distance
        # Only consider particles with a distance > 0.
        distance < sqrt(eps()) && return

        # Apply the same force to the solid particle
        # that the fluid particle experiences due to the soild particle.
        # Note that the same arguments are passed here as in fluid-solid interact!,
        # except that pos_diff has a flipped sign.
        #
        # In fluid-solid interaction, use the "hydrodynamic mass" of the solid particles
        # corresponding to the rest density of the fluid and not the material density.
        m_a = hydrodynamic_mass(particle_system, particle)
        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        rho_a = particle_density(v_particle_system, particle_system, particle)
        rho_b = particle_density(v_neighbor_system, neighbor_system, neighbor)
        rho_mean = (rho_a + rho_b) / 2

        # Use kernel from the fluid system in order to get the same force here in
        # solid-fluid interaction as for fluid-solid interaction.
        grad_kernel = smoothing_kernel_grad(neighbor_system, pos_diff, distance)

        # In fluid-solid interaction, use the "hydrodynamic pressure" of the solid particles
        # corresponding to the chosen boundary model.
        p_a = particle_pressure(v_particle_system, particle_system, particle)
        p_b = particle_pressure(v_neighbor_system, neighbor_system, neighbor)

        # Particle and neighbor (and corresponding systems and all corresponding quantities)
        # are switched in the following two calls.
        # This way, we obtain the exact same force as for the fluid-solid interaction,
        # but with a flipped sign (because `pos_diff` is flipped compared to fluid-solid).
        dv_boundary = pressure_acceleration(neighbor_system, particle_system, particle,
                                            m_b, m_a, p_b, p_a, rho_b, rho_a, pos_diff,
                                            distance, grad_kernel,
                                            neighbor_system.correction)

        dv_viscosity_ = dv_viscosity(neighbor_system, particle_system,
                                     v_neighbor_system, v_particle_system,
                                     neighbor, particle, pos_diff, distance,
                                     sound_speed, m_b, m_a, rho_mean)

        dv_particle = dv_boundary + dv_viscosity_

        for i in 1:ndims(particle_system)
            # Multiply `dv` (acceleration on fluid particle b) by the mass of
            # particle b to obtain the same force as for the fluid-solid interaction.
            # Divide by the material mass of particle a to obtain the acceleration
            # of solid particle a.
            total_dv[i] += dv[i, particle] +
                           dv_particle[i] * m_b / particle_system.mass[particle]
        end

        continuity_equation!(dv, v_particle_system, v_neighbor_system,
                             particle, neighbor, pos_diff, distance,
                             m_b, rho_a, rho_b,
                             particle_system, neighbor_system, grad_kernel)
    end

    #total_dv ./=nparticles(particle_system)
    for particle in each_moving_particle(particle_system)
        for i in 1:ndims(particle_system)
            dv[i, particle] += total_dv[i] / nparticles(particle_system)
        end
    end
    return dv
end

@inline function continuity_equation!(dv, v_particle_system, v_neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      m_b, rho_a, rho_b,
                                      particle_system::RigidSPHSystem,
                                      neighbor_system::FluidSystem,
                                      grad_kernel)
    return dv
end

@inline function continuity_equation!(dv, v_particle_system, v_neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      m_b, rho_a, rho_b,
                                      particle_system::RigidSPHSystem{<:BoundaryModelDummyParticles{ContinuityDensity}},
                                      neighbor_system::FluidSystem,
                                      grad_kernel)
    fluid_density_calculator = neighbor_system.density_calculator

    v_diff = current_velocity(v_particle_system, particle_system, particle) -
             current_velocity(v_neighbor_system, neighbor_system, neighbor)

    # Call the dummy BC version of the continuity equation
    continuity_equation!(dv, fluid_density_calculator, m_b, rho_a, rho_b, v_diff,
                         grad_kernel, particle)
end

# Solid-boundary interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system, neighborhood_search,
                   particle_system::RigidSPHSystem,
                   neighbor_system::Union{BoundarySPHSystem, RigidSPHSystem})

    # TODO continuity equation?
    return dv
end

# Solid-solid interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system, neighborhood_search,
                   particle_system::TotalLagrangianSPHSystem,
                   neighbor_system::TotalLagrangianSPHSystem)
    interact_solid_solid!(dv, neighborhood_search, particle_system, neighbor_system)
end

# Function barrier without dispatch for unit testing
@inline function interact_solid_solid!(dv, neighborhood_search, particle_system,
                                       neighbor_system)
    (; penalty_force) = particle_system

    # Different solids do not interact with each other (yet)
    if particle_system !== neighbor_system
        return dv
    end

    # Everything here is done in the initial coordinates
    system_coords = initial_coordinates(particle_system)
    neighbor_coords = initial_coordinates(neighbor_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    # For solid-solid interaction, this has to happen in the initial coordinates.
    for_particle_neighbor(particle_system, neighbor_system,
                          system_coords, neighbor_coords,
                          neighborhood_search) do particle, neighbor, initial_pos_diff,
                                                  initial_distance
        # Only consider particles with a distance > 0.
        initial_distance < sqrt(eps()) && return

        rho_a = particle_system.material_density[particle]
        rho_b = neighbor_system.material_density[neighbor]

        grad_kernel = smoothing_kernel_grad(particle_system, initial_pos_diff,
                                            initial_distance)

        m_b = neighbor_system.mass[neighbor]

        dv_particle = m_b *
                      (pk1_corrected(particle_system, particle) / rho_a^2 +
                       pk1_corrected(neighbor_system, neighbor) / rho_b^2) *
                      grad_kernel

        for i in 1:ndims(particle_system)
            dv[i, particle] += dv_particle[i]
        end

        calc_penalty_force!(dv, particle, neighbor, initial_pos_diff,
                            initial_distance, particle_system, penalty_force)

        # TODO continuity equation?
    end

    return dv
end

# Solid-fluid interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system, neighborhood_search,
                   particle_system::TotalLagrangianSPHSystem,
                   neighbor_system::FluidSystem)
    sound_speed = system_sound_speed(neighbor_system)

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

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
            dv[i, particle] += dv_particle[i] * m_b / particle_system.mass[particle]
        end

        continuity_equation!(dv, v_particle_system, v_neighbor_system,
                             particle, neighbor, pos_diff, distance,
                             m_b, rho_a, rho_b,
                             particle_system, neighbor_system, grad_kernel)
    end

    return dv
end

@inline function continuity_equation!(dv, v_particle_system, v_neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      m_b, rho_a, rho_b,
                                      particle_system::TotalLagrangianSPHSystem,
                                      neighbor_system::FluidSystem,
                                      grad_kernel)
    return dv
end

@inline function continuity_equation!(dv, v_particle_system, v_neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      m_b, rho_a, rho_b,
                                      particle_system::TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles{ContinuityDensity}},
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
                   particle_system::TotalLagrangianSPHSystem,
                   neighbor_system::BoundarySPHSystem)
    # TODO continuity equation?
    return dv
end

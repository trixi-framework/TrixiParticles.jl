# Solid-solid interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::TotalLagrangianSPHSystem,
                   neighbor_system::TotalLagrangianSPHSystem, semi;
                   integrate_tlsph=semi.integrate_tlsph[])
    # Skip interaction if TLSPH systems are integrated separately
    integrate_tlsph || return dv

    # Different solids do not interact with each other (yet)
    particle_system !== neighbor_system && return dv

    interact_solid_solid!(dv, v_particle_system, u_particle_system, particle_system, semi)
end

# Function barrier without dispatch for unit testing
@inline function interact_solid_solid!(dv, v, u, system, semi)
    (; penalty_force) = system

    # Everything here is done in the initial coordinates
    initial_coords = initial_coordinates(system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    # For solid-solid interaction, this has to happen in the initial coordinates.
    foreach_point_neighbor(system, system, initial_coords, initial_coords, semi;
                           points=each_moving_particle(system)) do particle,
                                                                   neighbor,
                                                                   initial_pos_diff,
                                                                   initial_distance
        # Only consider particles with a distance > 0.
        initial_distance < sqrt(eps()) && return

        rho_a = system.material_density[particle]
        rho_b = system.material_density[neighbor]

        grad_kernel = smoothing_kernel_grad(system, initial_pos_diff,
                                            initial_distance, particle)

        m_a = system.mass[particle]
        m_b = system.mass[neighbor]

        viscosity_tensor_ = viscosity_tensor(v, u, system, particle, neighbor)

        dv_particle = m_b *
                      (pk1_corrected(system, particle) / rho_a^2 +
                       pk1_corrected(system, neighbor) / rho_b^2 +
                       viscosity_tensor_) * grad_kernel

        for i in 1:ndims(system)
            @inbounds dv[i, particle] += dv_particle[i]
        end

        calc_penalty_force!(dv, particle, neighbor, initial_pos_diff,
                            initial_distance, system, m_a, m_b, rho_a, rho_b,
                            penalty_force)

        # TODO continuity equation?
    end

    return dv
end

@inline function viscosity_tensor(v, u, system, particle, neighbor)
    current_pos_diff = current_coords(u, system, particle) -
                       current_coords(u, system, neighbor)
    current_distance2 = dot(current_pos_diff, current_pos_diff)

    v_a = current_velocity(v, system, particle)
    v_b = current_velocity(v, system, neighbor)
    v_diff = v_a - v_b

    smoothing_length_particle = smoothing_length(system, particle)
    smoothing_length_neighbor = smoothing_length(system, neighbor)
    smoothing_length_average = (smoothing_length_particle + smoothing_length_neighbor) / 2

    alpha = 0.2
    epsilon = 0.01
    rho_a = system.material_density[particle]
    K = system.young_modulus / (3 * (1 - 2 * system.poisson_ratio))
    c = sqrt(K / rho_a)
    pi_ab = alpha * c * smoothing_length_average * rho_a * dot(v_diff, current_pos_diff) /
            (current_distance2 + epsilon^2 * smoothing_length_average^2)

    F = deformation_gradient(system, particle)

    return det(F) * pi_ab * inv(F)' / rho_a^2
    # return zero(SMatrix{ndims(system), ndims(system), eltype(system)})
end

# Solid-fluid interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::TotalLagrangianSPHSystem,
                   neighbor_system::FluidSystem, semi;
                   integrate_tlsph=semi.integrate_tlsph[])
    (; boundary_model) = particle_system

    # Skip interaction if TLSPH systems are integrated separately
    integrate_tlsph || return dv

    sound_speed = system_sound_speed(neighbor_system)

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    foreach_point_neighbor(particle_system, neighbor_system, system_coords, neighbor_coords,
                           semi;
                           points=each_moving_particle(particle_system)) do particle, neighbor,
                                                                            pos_diff, distance
        # Only consider particles with a distance > 0.
        distance < sqrt(eps()) && return

        # Apply the same force to the solid particle
        # that the fluid particle experiences due to the solid particle.
        # Note that the same arguments are passed here as in fluid-solid interact!,
        # except that pos_diff has a flipped sign.
        #
        # In fluid-solid interaction, use the "hydrodynamic mass" of the solid particles
        # corresponding to the rest density of the fluid and not the material density.
        m_a = hydrodynamic_mass(particle_system, particle)
        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        rho_a = current_density(v_particle_system, particle_system, particle)
        rho_b = current_density(v_neighbor_system, neighbor_system, neighbor)

        # Use kernel from the boundary model.
        # This should generally be the same as the kernel and smoothing length
        # of the fluid in order to get the same force here in solid-fluid interaction
        # as for fluid-solid interaction.
        # TODO this will not use corrections if the fluid uses corrections.
        grad_kernel = smoothing_kernel_grad(boundary_model, pos_diff, distance, particle)

        # In fluid-solid interaction, use the "hydrodynamic pressure" of the solid particles
        # corresponding to the chosen boundary model.
        p_a = current_pressure(v_particle_system, particle_system, particle)
        p_b = current_pressure(v_neighbor_system, neighbor_system, neighbor)

        # Particle and neighbor (and corresponding systems and all corresponding quantities)
        # are switched in the following two calls.
        # This way, we obtain the exact same force as for the fluid-solid interaction,
        # but with a flipped sign (because `pos_diff` is flipped compared to fluid-solid).
        dv_boundary = pressure_acceleration(neighbor_system, particle_system,
                                            neighbor, particle,
                                            m_b, m_a, p_b, p_a, rho_b, rho_a, pos_diff,
                                            distance, grad_kernel,
                                            neighbor_system.correction)

        dv_viscosity_ = dv_viscosity(neighbor_system, particle_system,
                                     v_neighbor_system, v_particle_system,
                                     neighbor, particle, pos_diff, distance,
                                     sound_speed, m_b, m_a, rho_a, rho_b, grad_kernel)

        dv_particle = dv_boundary + dv_viscosity_

        for i in 1:ndims(particle_system)
            # Multiply `dv` (acceleration on fluid particle b) by the mass of
            # particle b to obtain the same force as for the fluid-solid interaction.
            # Divide by the material mass of particle a to obtain the acceleration
            # of solid particle a.
            dv[i, particle] += dv_particle[i] * m_b / particle_system.mass[particle]
        end

        # continuity_equation!(dv, v_particle_system, v_neighbor_system,
        #                      particle, neighbor, pos_diff, distance,
        #                      m_b, rho_a, rho_b,
        #                      particle_system, neighbor_system, grad_kernel)
    end

    if particle_system.boundary_model isa BoundaryModelDummyParticles{ContinuityDensity}
        foreach_point_neighbor(particle_system, neighbor_system, system_coords, neighbor_coords,
                            semi;
                            points=eachparticle(particle_system)) do particle, neighbor,
                                                                                pos_diff, distance
            # Only consider particles with a distance > 0.
            distance < sqrt(eps()) && return

            # Apply the same force to the solid particle
            # that the fluid particle experiences due to the solid particle.
            # Note that the same arguments are passed here as in fluid-solid interact!,
            # except that pos_diff has a flipped sign.
            #
            # In fluid-solid interaction, use the "hydrodynamic mass" of the solid particles
            # corresponding to the rest density of the fluid and not the material density.
            m_a = hydrodynamic_mass(particle_system, particle)
            m_b = hydrodynamic_mass(neighbor_system, neighbor)

            rho_a = current_density(v_particle_system, particle_system, particle)
            rho_b = current_density(v_neighbor_system, neighbor_system, neighbor)


            v_a = current_velocity(v_particle_system, particle_system, particle)
            v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)

            # Use kernel from the boundary model.
            # This should generally be the same as the kernel and smoothing length
            # of the fluid in order to get the same force here in solid-fluid interaction
            # as for fluid-solid interaction.
            # TODO this will not use corrections if the fluid uses corrections.
            grad_kernel = smoothing_kernel_grad(boundary_model, pos_diff, distance, particle)

            continuity_equation!(dv, v_particle_system, v_neighbor_system,
                                particle, neighbor, pos_diff, distance,
                                m_b, rho_a, rho_b,
                                particle_system, neighbor_system, grad_kernel)

            shifting_continuity_equation!(dv, particle_shifting(neighbor_system), v_a, v_b, m_b, rho_a, rho_b,
                                        particle_system, neighbor_system, particle, neighbor,
                                        grad_kernel)

            density_diffusion!(dv, neighbor_system.density_diffusion, v_particle_system, particle, neighbor,
                        pos_diff, distance, m_b, rho_a, rho_b, particle_system, neighbor_system, # TODO neighbor_system
                        grad_kernel)
        end
    end

    return dv
end

@inline function density_diffusion!(dv, density_diffusion, v_particle_system, particle, neighbor,
                       pos_diff, distance, m_b, rho_a, rho_b, particle_system, neighbor_system,
                       grad_kernel)
    return dv
end

@inline function density_diffusion!(dv, density_diffusion, v_particle_system, particle, neighbor,
                       pos_diff, distance, m_b, rho_a, rho_b,
                       particle_system::TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles{ContinuityDensity}},
                       neighbor_system, grad_kernel)
    density_diffusion!(dv, density_diffusion, v_particle_system, particle, neighbor,
                       pos_diff, distance, m_b, rho_a, rho_b, neighbor_system,
                       grad_kernel)
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
                   v_neighbor_system, u_neighbor_system,
                   particle_system::TotalLagrangianSPHSystem,
                   neighbor_system::Union{BoundarySPHSystem, OpenBoundarySPHSystem}, semi;
                   integrate_tlsph=semi.integrate_tlsph[])
    # TODO continuity equation?
    return dv
end

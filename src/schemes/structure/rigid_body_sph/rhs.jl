# Structure-fluid interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::RigidSPHSystem,
                   neighbor_system::AbstractFluidSystem, semi)
    sound_speed = system_sound_speed(neighbor_system)

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    force_per_particle = particle_system.cache.force_per_particle
    set_zero!(force_per_particle)

    total_mass = sum(particle_system.mass)

    # Loop over all pairs of particles and neighbors within the kernel cutoff
    foreach_point_neighbor(particle_system, neighbor_system,
                           system_coords, neighbor_coords, semi;
                           points=each_integrated_particle(particle_system)) do particle,
                                                                                neighbor,
                                                                                pos_diff,
                                                                                distance
        # Only consider particles with a distance > 0.
        # See `src/general/smoothing_kernels.jl` for more details.
        distance^2 < eps(initial_smoothing_length(particle_system)^2) && return

        # Apply the same force to the structure particle that the fluid particle
        # experiences due to the structure particle.
        # In fluid-structure interaction, use the "hydrodynamic mass" of the structure
        # particles corresponding to the rest density of the fluid.
        m_a = hydrodynamic_mass(particle_system, particle)
        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        rho_a = current_density(v_particle_system, particle_system, particle)
        rho_b = current_density(v_neighbor_system, neighbor_system, neighbor)

        # Use the fluid kernel in order to get the same force as in
        # fluid-structure interaction.
        grad_kernel = smoothing_kernel_grad(neighbor_system, pos_diff, distance, neighbor)

        # In fluid-structure interaction, use the "hydrodynamic pressure" of the
        # structure particles corresponding to the chosen boundary model.
        p_a = current_pressure(v_particle_system, particle_system, particle)
        p_b = current_pressure(v_neighbor_system, neighbor_system, neighbor)

        # Particle and neighbor are switched in the following two calls.
        # This yields the opposite force of the fluid-structure interaction,
        # because `pos_diff` is flipped.
        dv_boundary = pressure_acceleration(neighbor_system, particle_system,
                                            neighbor, particle,
                                            m_b, m_a, p_b, p_a, rho_b, rho_a,
                                            pos_diff, distance, grad_kernel,
                                            neighbor_system.correction)

        dv_viscosity_ = dv_viscosity(neighbor_system, particle_system,
                                     v_neighbor_system, v_particle_system,
                                     neighbor, particle, pos_diff, distance,
                                     sound_speed, m_b, m_a, rho_a, rho_b,
                                     grad_kernel)

        dv_particle = dv_boundary + dv_viscosity_

        for i in 1:ndims(particle_system)
            force_per_particle[i, particle] += dv_particle[i] * m_b
        end

        continuity_equation!(dv, v_particle_system, v_neighbor_system,
                             particle, neighbor, pos_diff, distance,
                             m_b, rho_a, rho_b,
                             particle_system, neighbor_system, grad_kernel)
    end

    if total_mass > eps(eltype(total_mass))
        for i in 1:ndims(particle_system)
            total_force_i = zero(eltype(dv))

            for particle in each_integrated_particle(particle_system)
                total_force_i += force_per_particle[i, particle]
            end

            for particle in each_integrated_particle(particle_system)
                dv[i, particle] += total_force_i / total_mass
            end
        end
    end

    return dv
end

@inline function continuity_equation!(dv, v_particle_system, v_neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      m_b, rho_a, rho_b,
                                      particle_system::RigidSPHSystem,
                                      neighbor_system::AbstractFluidSystem,
                                      grad_kernel)
    return dv
end

@inline function continuity_equation!(dv, v_particle_system, v_neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      m_b, rho_a, rho_b,
                                      particle_system::RigidSPHSystem{<:BoundaryModelDummyParticles{ContinuityDensity}},
                                      neighbor_system::AbstractFluidSystem,
                                      grad_kernel)
    fluid_density_calculator = neighbor_system.density_calculator

    v_diff = current_velocity(v_particle_system, particle_system, particle) -
             current_velocity(v_neighbor_system, neighbor_system, neighbor)

    # Call the dummy boundary condition version of the continuity equation
    continuity_equation!(dv, fluid_density_calculator, m_b, rho_a, rho_b, v_diff,
                         grad_kernel, particle)
end

# Structure-structure and structure-boundary interactions are currently not modeled.
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::RigidSPHSystem,
                   neighbor_system::Union{AbstractStructureSystem,
                                          AbstractBoundarySystem,
                                          OpenBoundarySystem}, semi)
    return dv
end

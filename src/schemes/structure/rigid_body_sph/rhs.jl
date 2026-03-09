# Structure-fluid interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::RigidSPHSystem,
                   neighbor_system::AbstractFluidSystem, semi)
    sound_speed = system_sound_speed(neighbor_system)

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # Accumulate pairwise fluid forces per rigid particle first, then reduce them to a
    # single resultant force/torque for the rigid-body update below.
    force_per_particle = particle_system.force_per_particle
    set_zero!(force_per_particle)

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
                                     sound_speed, m_b, m_a, rho_b, rho_a,
                                     grad_kernel)

        dv_particle = dv_boundary + dv_viscosity_

        for i in 1:ndims(particle_system)
            # `pressure_acceleration`/`dv_viscosity` return acceleration-like pair contributions.
            # Multiply by the interacting fluid mass to recover the force on this rigid particle.
            force_per_particle[i, particle] += dv_particle[i] * m_b
        end

        continuity_equation!(dv, v_particle_system, v_neighbor_system,
                             particle, neighbor, pos_diff, distance,
                             m_b, rho_a, rho_b,
                             particle_system, neighbor_system, grad_kernel)
    end

    apply_resultant_force_and_torque!(dv, particle_system, semi)

    return dv
end

# Reduce the accumulated fluid forces to rigid-body resultants and apply the corresponding
# translational and rotational acceleration to every rigid particle.
function apply_resultant_force_and_torque!(dv, particle_system::RigidSPHSystem, semi)
    total_mass = particle_system.total_mass

    # Guard against degenerate systems and clear the cached rigid-body quantities as well.
    if total_mass <= eps(eltype(particle_system))
        particle_system.resultant_force[] = zero(particle_system.resultant_force[])
        particle_system.resultant_torque[] = zero(particle_system.resultant_torque[])
        particle_system.angular_acceleration_force[] = zero(particle_system.angular_acceleration_force[])
        return dv
    end

    # Reduce all pairwise forces to one net force and one net torque around the center of mass.
    total_force,
    total_torque = resultant_force_and_torque(particle_system,
                                              particle_system.force_per_particle,
                                              particle_system.relative_coordinates,
                                              Val(ndims(particle_system)))

    # Convert the rigid-body resultants into translational and angular accelerations.
    translational_acceleration = total_force / total_mass
    angular_acceleration_force = particle_system.inverse_inertia[] * total_torque
    particle_system.resultant_force[] = total_force
    particle_system.resultant_torque[] = total_torque
    particle_system.angular_acceleration_force[] = angular_acceleration_force

    @threaded semi for particle in each_integrated_particle(particle_system)
        relative_position = extract_svector(particle_system.relative_coordinates,
                                            particle_system,
                                            particle)
        # For rigid bodies, the instantaneous acceleration of a material point is
        # `a_com + alpha x r` in this force-driven part of the RHS.
        rotational_acceleration = angular_acceleration_cross_position(angular_acceleration_force,
                                                                      relative_position,
                                                                      Val(ndims(particle_system)))

        for i in 1:ndims(particle_system)
            dv[i, particle] += translational_acceleration[i] + rotational_acceleration[i]
        end
    end

    return dv
end

# Sum pairwise particle forces into a single net force and torque about the current
# center of mass of the rigid body.
function resultant_force_and_torque(particle_system::RigidSPHSystem, force_per_particle,
                                    relative_coordinates)
    particles = each_integrated_particle(particle_system)
    total_force,
    total_torque = mapreduce((x, y) -> x .+ y, particles) do particle
        particle_force = extract_svector(force_per_particle, particle_system, particle)
        relative_position = extract_svector(relative_coordinates, particle_system, particle)
        # Torque is taken about the current center of mass, using the particle's current
        # relative position inside the rigid body.
        particle_torque = cross(relative_position, particle_force)
        return (particle_force, particle_torque)
    end

    return total_force, total_torque
end

# Compute `alpha x r` for 2D rigid-body motion, where `alpha` is the scalar out-of-plane
# angular acceleration and `r` the in-plane particle position relative to the center of mass.
@inline function angular_acceleration_cross_position(angular_acceleration,
                                                     relative_position,
                                                     ::Val{2})
    # In 2D, angular acceleration is a scalar in the out-of-plane direction.
    return SVector(-angular_acceleration * relative_position[2],
                   angular_acceleration * relative_position[1])
end

# Compute `alpha x r` for 3D rigid-body motion.
@inline function angular_acceleration_cross_position(angular_acceleration,
                                                     relative_position,
                                                     ::Val{3})
    return cross(angular_acceleration, relative_position)
end

# Default rigid boundary models keep density fixed, so structure-fluid coupling does not
# contribute to a density RHS entry.
@inline function continuity_equation!(dv, v_particle_system, v_neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      m_b, rho_a, rho_b,
                                      particle_system::RigidSPHSystem,
                                      neighbor_system::AbstractFluidSystem,
                                      grad_kernel)
    # Most rigid boundary models keep their density fixed, so no continuity update is needed.
    return dv
end

# Dummy-particle rigid boundaries with `ContinuityDensity` reuse the fluid-compatible
# density update for the rigid particle.
@inline function continuity_equation!(dv, v_particle_system, v_neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      m_b, rho_a, rho_b,
                                      particle_system::RigidSPHSystem{<:BoundaryModelDummyParticles{ContinuityDensity}},
                                      neighbor_system::AbstractFluidSystem,
                                      grad_kernel)
    # Dummy rigid particles with `ContinuityDensity` reuse the fluid-side density update.
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

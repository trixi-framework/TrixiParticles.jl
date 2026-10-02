# Computes the forces that particles in `particle_system` experience from particles
# in `neighbor_system` and updates `dv` accordingly.
# It takes into account pressure forces, viscosity, and for `ContinuityDensity` updates the density
# using the continuity equation.
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::ImplicitIncompressibleSPHSystem,
                   neighbor_system, semi)
    sound_speed = system_sound_speed(particle_system) #TODO
    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # For `distance == 0`, the analytical gradient is zero, but the unsafe gradient
    # and the density diffusion divide by zero.
    # To account for rounding errors, we check if `distance` is almost zero.
    # Since the coordinates are in the order of the smoothing length `h`, `distance^2` is in
    # the order of `h^2`, so we need to check `distance < sqrt(eps(h^2))`.
    # Note that `sqrt(eps(h^2)) != eps(h)`.
    compact_support_ = compact_support(particle_system, neighbor_system)
    almostzero = interaction_zero_distance(particle_system, neighbor_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    foreach_point_neighbor(particle_system, neighbor_system,
                           system_coords, neighbor_system_coords, semi;
                           points=each_integrated_particle(particle_system)) do particle,
                                                                                neighbor,
                                                                                pos_diff,
                                                                                distance
        # Skip neighbors with the same position because the kernel gradient is zero.
        # Note that `return` only exits the closure, i.e., skips the current neighbor.
        skip_fluid_pair(particle_system, distance, compact_support_, almostzero) && return

        # Now that we know that `distance` is not zero, we can safely call the unsafe
        # version of the kernel gradient to avoid redundant zero checks.
        grad_kernel = smoothing_kernel_grad_unsafe(particle_system, pos_diff,
                                                   distance, particle)

        # `foreach_point_neighbor` makes sure that `particle` and `neighbor` are
        # in bounds of the respective system. For performance reasons, we use `@inbounds`
        # in this hot loop to avoid bounds checking when extracting particle quantities.
        rho_a = @inbounds current_density(v_particle_system, particle_system, particle)
        rho_b = @inbounds current_density(v_neighbor_system, neighbor_system, neighbor)

        v_a = @inbounds current_velocity(v_particle_system, particle_system, particle)
        v_b = @inbounds current_velocity(v_neighbor_system, neighbor_system, neighbor)

        m_a = @inbounds hydrodynamic_mass(particle_system, particle)
        m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)

        p_a = @inbounds current_pressure(v_particle_system, particle_system, particle)
        # The following call is equivalent to
        #     `p_b = current_pressure(v_neighbor_system, neighbor_system, neighbor)`
        # For boundaries and structures using `PressureMirroring`, this returns
        # `p_b = p_a`, which is the pressure of the fluid particle.
        p_b = @inbounds neighbor_pressure(v_neighbor_system, neighbor_system,
                                          neighbor, p_a)

        dv_particle = @inbounds fluid_pair_acceleration(particle_system, neighbor_system,
                                                        v_particle_system,
                                                        v_neighbor_system,
                                                        particle, neighbor,
                                                        m_a, m_b, p_a, p_b, rho_a, rho_b,
                                                        v_a, v_b, pos_diff, distance,
                                                        sound_speed, grad_kernel, nothing)

        for i in 1:ndims(particle_system)
            @inbounds dv[i, particle] += dv_particle[i]
        end
    end
    return dv
end

# Computes the forces that particles in `particle_system` experience from particles
# in `neighbor_system` and updates `dv` accordingly.
# It takes into account pressure forces, viscosity, and for `ContinuityDensity` updates
# the density using the continuity equation.
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::WeaklyCompressibleSPHSystem, neighbor_system, semi;
                   eachparticle=each_integrated_particle(particle_system),
                   kwargs...)
    (; density_calculator, correction) = particle_system

    sound_speed = system_sound_speed(particle_system)

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)
    neighborhood_search = get_neighborhood_search(particle_system, neighbor_system, semi)
    backend = semi.parallelization_backend

    # For `distance == 0`, the analytical gradient is zero, but the unsafe gradient divides
    # by zero. To account for rounding errors, we check if `distance` is almost zero.
    # Since the coordinates are in the order of the compact support `c`, `distance^2` is in
    # the order of `c^2`, so we need to check `distance < sqrt(eps(c^2))`.
    # Note that `sqrt(eps(c^2)) != eps(c)`.
    compact_support_ = compact_support(particle_system, neighbor_system)
    almostzero = interaction_zero_distance(particle_system, neighbor_system)

    @threaded semi for particle in eachparticle
        # We are looping over the particles of `particle_system`, so it is guaranteed
        # that `particle` is in bounds of `particle_system`.
        m_a = @inbounds hydrodynamic_mass(particle_system, particle)
        p_a = @inbounds current_pressure(v_particle_system, particle_system, particle)

        v_a = @inbounds current_velocity(v_particle_system, particle_system, particle)
        rho_a = @inbounds current_density(v_particle_system, particle_system, particle)

        # Accumulate the RHS contributions over all neighbors before writing to `dv`,
        # to reduce the number of memory writes.
        init = (zero(v_a), zero(rho_a))

        # Loop over all neighbors within the kernel cutoff.
        # Make sure that the returned names `dv_particle_` and `drho_particle_`
        # are not used inside the closure to avoid allocations.
        (dv_particle_,
         drho_particle_) = @inbounds mapreduce_neighbor(sum_interaction_contributions,
                                                        system_coords,
                                                        neighbor_system_coords,
                                                        neighborhood_search,
                                                        backend, particle;
                                                        init) do particle, neighbor,
                                                                 pos_diff, distance
            # Skip neighbors with the same position because the kernel gradient is zero.
            # Note that `return` only exits the closure, i.e., skips the current neighbor.
            skip_fluid_pair(particle_system, distance, compact_support_, almostzero) &&
                return init

            # Now that we know that `distance` is not zero, we can safely call the unsafe
            # version of the kernel gradient to avoid redundant zero checks.
            grad_kernel = smoothing_kernel_grad_unsafe(particle_system, pos_diff,
                                                       distance, particle)

            # `foreach_neighbor` makes sure that `neighbor` is in bounds of `neighbor_system`
            m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)
            v_b = @inbounds current_velocity(v_neighbor_system, neighbor_system, neighbor)
            rho_b = @inbounds current_density(v_neighbor_system, neighbor_system, neighbor)

            # The following call is equivalent to
            #     `p_b = current_pressure(v_neighbor_system, neighbor_system, neighbor)`
            # For boundaries and structures using `PressureMirroring`, this returns
            # `p_b = p_a`, which is the pressure of the fluid particle.
            p_b = @inbounds neighbor_pressure(v_neighbor_system, neighbor_system,
                                              neighbor, p_a)

            dv_particle = @inbounds physical_fluid_pair_acceleration(particle_system,
                                                                     neighbor_system,
                                                                     v_particle_system,
                                                                     v_neighbor_system,
                                                                     particle, neighbor,
                                                                     m_a, m_b, p_a, p_b,
                                                                     rho_a,
                                                                     rho_b,
                                                                     v_a, v_b, pos_diff,
                                                                     distance,
                                                                     sound_speed,
                                                                     grad_kernel,
                                                                     correction)

            # Extra terms in the momentum equation when using a shifting technique
            dv_particle = @inbounds add_dv_shifting(dv_particle,
                                                    shifting_technique(particle_system),
                                                    particle_system, neighbor_system,
                                                    v_particle_system, v_neighbor_system,
                                                    particle, neighbor, m_a, m_b, rho_a,
                                                    rho_b, v_a, v_b, pos_diff, distance,
                                                    grad_kernel, correction)

            drho_particle = zero(rho_a)

            # TODO If variable smoothing_length is used, this should use the neighbor smoothing length
            # Propagate `@inbounds` to the continuity equation, which accesses particle data
            drho_particle = @inbounds add_continuity_equation(drho_particle,
                                                              density_calculator,
                                                              particle_system,
                                                              neighbor_system,
                                                              particle, neighbor,
                                                              pos_diff, distance,
                                                              m_b, rho_a, rho_b, v_a, v_b,
                                                              grad_kernel)

            return dv_particle, drho_particle
        end

        for i in eachindex(dv_particle_)
            @inbounds dv[i, particle] += dv_particle_[i]
        end
        @inbounds write_drho_particle!(dv, density_calculator, drho_particle_, particle)
    end

    return dv
end

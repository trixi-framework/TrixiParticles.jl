# Computes the forces that particles in `particle_system` experience from particles
# in `neighbor_system` and updates `dv` accordingly.
# It takes into account pressure forces, viscosity, and for `ContinuityDensity` updates
# the density using the continuity equation.
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::WeaklyCompressibleSPHSystem, neighbor_system, semi)
    (; density_calculator, correction) = particle_system

    sound_speed = system_sound_speed(particle_system)

    surface_tension_a = surface_tension_model(particle_system)
    surface_tension_b = surface_tension_model(neighbor_system)

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)
    neighborhood_search = get_neighborhood_search(particle_system, neighbor_system, semi)

    # For `distance == 0`, the analytical gradient is zero, but the unsafe gradient divides
    # by zero. To account for rounding errors, we check if `distance` is almost zero.
    # Since the coordinates are in the order of the compact support `c`, `distance^2` is in
    # the order of `c^2`, so we need to check `distance < sqrt(eps(c^2))`.
    # Note that `sqrt(eps(c^2)) != eps(c)`.
    compact_support_ = compact_support(particle_system, neighbor_system)
    almostzero = sqrt(eps(compact_support_^2))

    @threaded semi for particle in each_integrated_particle(particle_system)
        # We are looping over the particles of `particle_system`, so it is guaranteed
        # that `particle` is in bounds of `particle_system`.
        m_a = @inbounds hydrodynamic_mass(particle_system, particle)
        p_a = @inbounds current_pressure(v_particle_system, particle_system, particle)

        # In 3D, this function can combine velocity and density load into one wide load,
        # which gives a significant speedup on GPUs.
        v_a, rho_a = @inbounds velocity_and_density(v_particle_system, particle_system,
                                                    particle)

        # Accumulate the RHS contributions over all neighbors before writing to `dv`,
        # to reduce the number of memory writes.
        # Note that we need a `Ref` in order to be able to update these variables
        # inside the closure in the `foreach_neighbor` loop.
        dv_particle = Ref(zero(v_a))
        drho_particle = Ref(zero(rho_a))

        # Loop over all neighbors within the kernel cutoff
        @inbounds PointNeighbors.foreach_neighbor(system_coords, neighbor_system_coords,
                                                  neighborhood_search,
                                                  particle) do particle, neighbor,
                                                               pos_diff, distance
            # Skip neighbors with the same position because the kernel gradient is zero.
            # Note that `return` only exits the closure, i.e., skips the current neighbor.
            skip_zero_distance(particle_system, distance, almostzero) && return

            # Now that we know that `distance` is not zero, we can safely call the unsafe
            # version of the kernel gradient to avoid redundant zero checks.
            grad_kernel = smoothing_kernel_grad_unsafe(particle_system, pos_diff,
                                                       distance, particle)

            # `foreach_neighbor` makes sure that `neighbor` is in bounds of `neighbor_system`
            m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)
            v_b, rho_b = @inbounds velocity_and_density(v_neighbor_system, neighbor_system,
                                                        neighbor)
            rho_mean = (rho_a + rho_b) / 2
            vdiff = v_a - v_b

            # The following call is equivalent to
            #     `p_b = current_pressure(v_neighbor_system, neighbor_system, neighbor)`
            # Only when the neighbor system is a `WallBoundarySystem`
            # or a `TotalLagrangianSPHSystem` with the boundary model `PressureMirroring`,
            # this will return `p_b = p_a`, which is the pressure of the fluid particle.
            p_b = @inbounds neighbor_pressure(v_neighbor_system, neighbor_system,
                                              neighbor, p_a)

            # For `ContinuityDensity` without correction, this is equivalent to
            # dv_pressure = -m_b * (p_a + p_b) / (rho_a * rho_b) * grad_kernel
            dv_pressure = pressure_acceleration(particle_system, neighbor_system,
                                                particle, neighbor,
                                                m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff,
                                                distance, grad_kernel, correction)

            # Propagate `@inbounds` to the viscosity function, which accesses particle data
            dv_viscosity_ = @inbounds dv_viscosity!(dv_particle,
                                                    particle_system, neighbor_system,
                                                    v_particle_system, v_neighbor_system,
                                                    particle, neighbor, pos_diff, distance,
                                                    sound_speed, m_a, m_b, rho_a, rho_b,
                                                    grad_kernel)

            # Extra terms in the momentum equation when using a shifting technique
            @inbounds dv_shifting!(dv_particle, shifting_technique(particle_system),
                                   particle_system, neighbor_system,
                                   v_particle_system, v_neighbor_system,
                                   particle, neighbor, m_a, m_b, rho_a, rho_b,
                                   pos_diff, distance, grad_kernel, correction)

            # TODO surface_tension_correction
            @inbounds surface_tension_force!(dv_particle, surface_tension_a,
                                             surface_tension_b,
                                             particle_system, neighbor_system,
                                             particle, neighbor, pos_diff, distance,
                                             rho_a, rho_b, grad_kernel)

            @inbounds adhesion_force!(dv_particle, surface_tension_a, particle_system,
                                      neighbor_system,
                                      particle, neighbor, pos_diff, distance)

            # Determine correction factors.
            # This can usually be ignored, as these are all 1 when no correction is used.
            (viscosity_correction, pressure_correction,
            surface_tension_correction) = free_surface_correction(correction, particle_system,
                                                                  rho_mean)

            # Accumulate contributions over all neighbors
            dv_particle[] += dv_pressure * pressure_correction +
                             dv_viscosity_ * viscosity_correction

            # TODO If variable smoothing_length is used, this should use the neighbor smoothing length
            # Propagate `@inbounds` to the continuity equation, which accesses particle data
            @inbounds continuity_equation!(drho_particle, density_calculator, particle_system,
                                           neighbor_system, v_particle_system,
                                           v_neighbor_system, particle, neighbor,
                                           pos_diff, distance, m_b, rho_a, rho_b, vdiff,
                                           grad_kernel)
        end

        for i in eachindex(dv_particle[])
            @inbounds dv[i, particle] += dv_particle[][i]
        end
        @inbounds dv[end, particle] += drho_particle[]
    end

    return dv
end

@propagate_inbounds function neighbor_pressure(v_neighbor_system, neighbor_system,
                                               neighbor, p_a)
    return current_pressure(v_neighbor_system, neighbor_system, neighbor)
end

@inline function neighbor_pressure(v_neighbor_system,
                                   neighbor_system::WallBoundarySystem{<:BoundaryModelDummyParticles{PressureMirroring}},
                                   neighbor, p_a)
    return p_a
end

@propagate_inbounds function velocity_and_density(v, system, particle)
    # For other systems, fall back to the default implementation
    return velocity_and_density(v, nothing, system, particle)
end

@propagate_inbounds function velocity_and_density(v, system::WeaklyCompressibleSPHSystem,
                                                  particle)
    (; density_calculator) = system

    return velocity_and_density(v, density_calculator, system, particle)
end

@propagate_inbounds function velocity_and_density(v, _, system, particle)
    v_particle = current_velocity(v, system, particle)
    rho_particle = current_density(v, system, particle)

    return v_particle, rho_particle
end

@inline function velocity_and_density(v, ::ContinuityDensity,
                                      ::WeaklyCompressibleSPHSystem{3}, particle)
    vrho_a = vloada(Vec{4, eltype(v)}, pointer(v, 4 * (particle - 1) + 1))
    a, b, c, d = Tuple(vrho_a)
    v_particle = SVector(a, b, c)
    rho_particle = d

    return v_particle, rho_particle
end

# Computes the forces that particles in `particle_system` experience from particles
# in `neighbor_system` and updates `dv` accordingly.
# It takes into account pressure forces, viscosity, and for `ContinuityDensity` updates
# the density using the continuity equation.
using LoopVectorization
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::WeaklyCompressibleSPHSystem, neighbor_system, semi)
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
    almostzero = eps(compact_support_^2)

    search_radius2 = compact_support_^2

    @threaded semi for particle in each_integrated_particle(particle_system)
        # We are looping over the particles of `particle_system`, so it is guaranteed
        # that `particle` is in bounds of `particle_system`.
        coords_a = @inbounds extract_svector(system_coords, Val(ndims(neighborhood_search)),
                                        particle)
        m_a = @inbounds hydrodynamic_mass(particle_system, particle)
        p_a = @inbounds current_pressure(v_particle_system, particle_system, particle)

        v_a = @inbounds current_velocity(v_particle_system, particle_system, particle)
        rho_a = @inbounds current_density(v_particle_system, particle_system, particle)

        # Accumulate the RHS contributions over all neighbors before writing to `dv`,
        # to reduce the number of memory writes.
        # Note that we need a `Ref` in order to be able to update these variables
        # inside the closure in the `foreach_neighbor` loop.
        dv_particle = zero(v_a)
        drho_particle = zero(rho_a)

        neighbors = neighborhood_search.neighbor_lists[particle]

        # cell = PointNeighbors.cell_coords(coords_a, neighborhood_search)
        # for neighbor_cell_ in PointNeighbors.neighboring_cells(cell, neighborhood_search)
        #     neighbor_cell = Tuple(neighbor_cell_)
        #     neighbors = @inbounds PointNeighbors.points_in_cell(neighbor_cell, neighborhood_search)

        for neighbor_ in eachindex(neighbors)
            neighbor = @inbounds neighbors[neighbor_]

            coords_b = @inbounds extract_svector(neighbor_system_coords,
                                                    Val(ndims(neighborhood_search)), neighbor)

            pos_diff = coords_a - coords_b
            distance2 = dot(pos_diff, pos_diff)

            # Only for grid NHS:
            # distance2 > search_radius2 && continue

            # Skip neighbors with the same position because the kernel gradient is zero.
            # Note that `return` only exits the closure, i.e., skips the current neighbor.
            # skip_zero_distance(particle_system) && distance2 < almostzero && continue
            distance = sqrt(distance2)

            # Now that we know that `distance` is not zero, we can safely call the unsafe
            # version of the kernel gradient to avoid redundant zero checks.
            grad_kernel = smoothing_kernel_grad_unsafe(particle_system, pos_diff,
                                                       distance, particle)

            # `foreach_neighbor` makes sure that `neighbor` is in bounds of `neighbor_system`
            m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)
            v_b = @inbounds current_velocity(v_neighbor_system, neighbor_system, neighbor)
            rho_b = @inbounds current_density(v_neighbor_system, neighbor_system, neighbor)
            p_b = @inbounds current_pressure(v_neighbor_system, neighbor_system, neighbor)

            # For `ContinuityDensity` without correction, this is equivalent to
            dv_pressure = -m_b * (p_a + p_b) / (rho_a * rho_b) * grad_kernel
            dv_particle += dv_pressure

            # Artificial viscosity
            # v_ab ⋅ r_ab
            v_diff = v_a - v_b
            vr = dot(v_diff, pos_diff)

            if vr < 0
                h_a = smoothing_length(particle_system, particle)
                h_b = smoothing_length(neighbor_system, neighbor)
                h = (h_a + h_b) / 2

                rho_mean = (rho_a + rho_b) / 2
                (; alpha, beta, epsilon) = particle_system.viscosity

                # Since this is one of the most performance critical functions, using fast divisions
                # here gives a significant speedup on GPUs.
                # See the docs page "Development" for more details on `div_fast`.
                mu = h * vr / (distance^2 + epsilon * h^2)
                c = sound_speed
                # TODO why is m_b inside the `div_fast` faster on H100 than `m_b * div_fast(...)`?
                dv_viscosity = (m_b * alpha * c * mu + m_b * beta * mu^2) / rho_mean *
                            grad_kernel
                dv_particle += dv_viscosity
            end

            drho_particle += rho_a / rho_b * m_b * dot(v_diff, grad_kernel)
        end
    # end

        for i in eachindex(dv_particle)
            @inbounds dv[i, particle] += dv_particle[i]
        end
        @inbounds dv[end, particle] += drho_particle
    end

    return dv
end

function interact_old!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::WeaklyCompressibleSPHSystem, neighbor_system, semi)
    (; density_calculator, correction) = particle_system

    sound_speed = system_sound_speed(particle_system)

    surface_tension_a = surface_tension_model(particle_system)
    surface_tension_b = surface_tension_model(neighbor_system)

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
    almostzero = sqrt(eps(compact_support_^2))

    @threaded semi for particle in each_integrated_particle(particle_system)
        # We are looping over the particles of `particle_system`, so it is guaranteed
        # that `particle` is in bounds of `particle_system`.
        m_a = @inbounds hydrodynamic_mass(particle_system, particle)
        p_a = @inbounds current_pressure(v_particle_system, particle_system, particle)

        v_a = @inbounds current_velocity(v_particle_system, particle_system, particle)
        rho_a = @inbounds current_density(v_particle_system, particle_system, particle)

        # Accumulate the RHS contributions over all neighbors before writing to `dv`,
        # to reduce the number of memory writes.
        # Note that we need a `Ref` in order to be able to update these variables
        # inside the closure in the `foreach_neighbor` loop.
        dv_particle = Ref(zero(v_a))
        drho_particle = Ref(zero(rho_a))

        # Loop over all neighbors within the kernel cutoff
        @inbounds foreach_neighbor(system_coords, neighbor_system_coords,
                                   neighborhood_search, backend,
                                   particle) do particle, neighbor, pos_diff, distance
            # Skip neighbors with the same position because the kernel gradient is zero.
            # Note that `return` only exits the closure, i.e., skips the current neighbor.
            skip_zero_distance(particle_system) && distance < almostzero && return

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
            # Only when the neighbor system is a `WallBoundarySystem`
            # or a `TotalLagrangianSPHSystem` with the boundary model `PressureMirroring`,
            # this will return `p_b = p_a`, which is the pressure of the fluid particle.
            p_b = @inbounds neighbor_pressure(v_neighbor_system, neighbor_system,
                                              neighbor, p_a)

            # Determine correction factors.
            # This can usually be ignored, as these are all 1 when no correction is used.
            (viscosity_correction, pressure_correction,
             surface_tension_correction) = free_surface_correction(correction,
                                                                   particle_system,
                                                                   rho_a, rho_b)

            # For `ContinuityDensity` without correction, this is equivalent to
            # dv_pressure = -m_b * (p_a + p_b) / (rho_a * rho_b) * grad_kernel
            dv_pressure = pressure_acceleration(particle_system, neighbor_system,
                                                particle, neighbor,
                                                m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff,
                                                distance, grad_kernel, correction)
            dv_particle[] += dv_pressure * pressure_correction

            # Propagate `@inbounds` to the viscosity function, which accesses particle data
            @inbounds dv_viscosity!(dv_particle, particle_system, neighbor_system,
                                    v_particle_system, v_neighbor_system,
                                    particle, neighbor, pos_diff, distance,
                                    sound_speed, m_a, m_b, rho_a, rho_b,
                                    v_a, v_b, grad_kernel, viscosity_correction)

            # Extra terms in the momentum equation when using a shifting technique
            @inbounds dv_shifting!(dv_particle, shifting_technique(particle_system),
                                   particle_system, neighbor_system,
                                   v_particle_system, v_neighbor_system,
                                   particle, neighbor, m_a, m_b, rho_a, rho_b, v_a, v_b,
                                   pos_diff, distance, grad_kernel, correction)

            @inbounds surface_tension_force!(dv_particle,
                                             surface_tension_a, surface_tension_b,
                                             particle_system, neighbor_system,
                                             particle, neighbor, pos_diff, distance,
                                             rho_a, rho_b, grad_kernel,
                                             surface_tension_correction)

            @inbounds adhesion_force!(dv_particle, surface_tension_a, particle_system,
                                      neighbor_system,
                                      particle, neighbor, pos_diff, distance)

            # TODO If variable smoothing_length is used, this should use the neighbor smoothing length
            # Propagate `@inbounds` to the continuity equation, which accesses particle data
            @inbounds continuity_equation!(drho_particle, density_calculator,
                                           particle_system, neighbor_system,
                                           particle, neighbor, pos_diff, distance,
                                           m_b, rho_a, rho_b, v_a, v_b, grad_kernel)
        end

        for i in eachindex(dv_particle[])
            @inbounds dv[i, particle] += dv_particle[][i]
        end
        @inbounds write_drho_particle!(dv, density_calculator, drho_particle, particle)
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

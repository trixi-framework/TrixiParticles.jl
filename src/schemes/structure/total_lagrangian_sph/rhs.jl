# Structure-structure interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::TotalLagrangianSPHSystem,
                   neighbor_system::TotalLagrangianSPHSystem, semi;
                   integrate_tlsph=semi.integrate_tlsph[],
                   eachparticle=each_integrated_particle(particle_system))
    # Different structures do not interact with each other (yet)
    particle_system === neighbor_system || return dv

    # Skip interaction if TLSPH systems are integrated separately
    integrate_tlsph || return dv

    interact_structure_structure!(dv, v_particle_system, particle_system, semi;
                                  eachparticle)
end

# Function barrier without dispatch for unit testing
@inline function interact_structure_structure!(dv, v_system, system, semi;
                                               eachparticle=each_integrated_particle(system))
    (; penalty_force) = system

    # Everything here is done in the initial coordinates
    system_coords = initial_coordinates(system)
    neighborhood_search = get_neighborhood_search(system, semi)
    backend = semi.parallelization_backend

    # For `distance == 0`, the analytical gradient is zero, but the unsafe gradient
    # and the density diffusion divide by zero.
    # To account for rounding errors, we check if `distance` is almost zero.
    # Since the coordinates are in the order of the smoothing length `h`, `distance^2` is in
    # the order of `h^2`, so we need to check `distance < sqrt(eps(h^2))`.
    # Note that `sqrt(eps(h^2)) != eps(h)`.
    h = initial_smoothing_length(system)
    almostzero = sqrt(eps(h^2))

    @threaded semi for particle in eachparticle
        # We are looping over the particles of `system`, so it is guaranteed
        # that `particle` is in bounds of `system`.
        m_a = @inbounds system.mass[particle]
        rho_a = @inbounds system.material_density[particle]
        # PK1 / rho^2
        pk1_rho2_a = @inbounds pk1_rho2(system, particle)
        current_coords_a = @inbounds current_coords(system, particle)
        F_a = @inbounds deformation_gradient(system, particle)

        # Accumulate the RHS contributions over all neighbors before writing to `dv`
        # to reduce the number of memory writes.
        # Note that we need a `Ref` in order to be able to update these variables
        # inside the closure in the `foreach_neighbor` loop.
        dv_particle = Ref(zero(current_coords_a))

        # Loop over all neighbors within the kernel cutoff
        @inbounds foreach_neighbor(system_coords, system_coords,
                                   neighborhood_search, backend,
                                   particle) do particle, neighbor,
                                                initial_pos_diff, initial_distance
            # Skip neighbors with the same position because the kernel gradient is zero.
            # Note that `return` only exits the closure, i.e., skips the current neighbor.
            skip_zero_distance(system) && initial_distance < almostzero && return

            # Now that we know that `distance` is not zero, we can safely call the unsafe
            # version of the kernel gradient to avoid redundant zero checks.
            grad_kernel = smoothing_kernel_grad_unsafe(system, initial_pos_diff,
                                                       initial_distance, particle)

            rho_b = @inbounds system.material_density[neighbor]
            m_b = @inbounds system.mass[neighbor]
            # PK1 / rho^2
            pk1_rho2_b = @inbounds pk1_rho2(system, neighbor)
            current_coords_b = @inbounds current_coords(system, neighbor)

            # The compiler is smart enough to optimize this away if no penalty force is used
            F_b = @inbounds deformation_gradient(system, neighbor)

            current_pos_diff_ = current_coords_a - current_coords_b
            # In mixed-precision simulations, convert from `coordinates_eltype(system)`
            # to `eltype(system)` immediately after computing the difference.
            current_pos_diff = convert.(eltype(system), current_pos_diff_)
            current_distance = norm(current_pos_diff)

            dv_particle[] += m_b * (pk1_rho2_a + pk1_rho2_b) * grad_kernel

            @inbounds dv_penalty_force!(dv_particle, penalty_force, particle, neighbor,
                                        initial_pos_diff, initial_distance,
                                        current_pos_diff, current_distance,
                                        system, m_a, m_b, rho_a, rho_b, F_a, F_b)

            @inbounds dv_viscosity_tlsph!(dv_particle, system, v_system, particle, neighbor,
                                          current_pos_diff, current_distance,
                                          m_a, m_b, rho_a, rho_b, F_a, grad_kernel)
        end

        for i in 1:ndims(system)
            @inbounds dv[i, particle] += dv_particle[][i]
        end

        # TODO continuity equation for boundary model with `ContinuityDensity`?
    end

    return dv
end

# Structure-fluid interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::TotalLagrangianSPHSystem,
                   neighbor_system::AbstractFluidSystem, semi;
                   integrate_tlsph=semi.integrate_tlsph[],
                   eachparticle=each_integrated_particle(particle_system))
    # Skip interaction if TLSPH systems are integrated separately
    integrate_tlsph || return dv

    return interact_structure_fluid!(dv, v_particle_system, u_particle_system,
                                     v_neighbor_system, u_neighbor_system,
                                     particle_system, neighbor_system, semi; eachparticle)
end

# Structure-boundary interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::TotalLagrangianSPHSystem,
                   neighbor_system::Union{WallBoundarySystem, OpenBoundarySystem}, semi;
                   integrate_tlsph=semi.integrate_tlsph[],
                   eachparticle=each_integrated_particle(particle_system))
    # TODO continuity equation?
    return dv
end

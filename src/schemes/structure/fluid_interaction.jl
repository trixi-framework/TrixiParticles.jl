# Structure-fluid coupling shared by TLSPH and rigid-body systems.
# Traversal labels are a=s, b=f, pos_diff=x_s-x_f; evaluate the fluid force with -pos_diff.
# Mechanical-work calculations override eachparticle to include clamped-particle loads.
function interact_structure_fluid!(dv, v_particle_system, u_particle_system,
                                   v_neighbor_system, u_neighbor_system,
                                   particle_system,
                                   neighbor_system::AbstractFluidSystem, semi;
                                   eachparticle=each_integrated_particle(particle_system))
    sound_speed = system_sound_speed(neighbor_system)
    correction = system_correction(neighbor_system)

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)
    neighborhood_search = get_neighborhood_search(particle_system, neighbor_system, semi)
    backend = semi.parallelization_backend

    compact_support_ = compact_support(neighbor_system, particle_system)
    h = initial_smoothing_length(neighbor_system)
    zero_distance_squared = eps(typeof(h)) * h^2

    @threaded semi for particle in eachparticle
        # In fluid-structure interaction, use the "hydrodynamic mass" of the structure particles
        # corresponding to the rest density of the fluid and not the material density.
        m_a = @inbounds hydrodynamic_mass(particle_system, particle)
        rho_a = @inbounds current_density(v_particle_system, particle_system, particle)
        v_a = @inbounds current_velocity(v_particle_system, particle_system, particle)

        init = (zero(v_a), zero(rho_a))

        # Keep the returned name out of the closure to avoid allocations.
        (force_particle_,
         drho_particle_) = @inbounds mapreduce_neighbor(sum_interaction_contributions,
                                                        system_coords,
                                                        neighbor_coords,
                                                        neighborhood_search, backend,
                                                        particle;
                                                        init) do particle,
                                                                 neighbor,
                                                                 pos_diff,
                                                                 distance
            m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)
            rho_b = @inbounds current_density(v_neighbor_system, neighbor_system, neighbor)
            v_b = @inbounds current_velocity(v_neighbor_system, neighbor_system, neighbor)

            # Boundary density can use pairs outside fluid support, so evaluate it
            # before the momentum cutoff and preserve its rate on early exit.
            drho_particle = @inbounds add_continuity_equation(zero(rho_a),
                                                              particle_system,
                                                              neighbor_system,
                                                              particle, neighbor, pos_diff,
                                                              distance, m_b, rho_a, rho_b,
                                                              v_a, v_b)

            # Corrected coincident-particle gradients can be finite and nonzero.
            if distance > compact_support_ ||
               (skip_zero_distance(neighbor_system) && distance^2 < zero_distance_squared)
                return zero(v_a), drho_particle
            end

            # Corrected gradients need not be odd: evaluate grad_f W(x_f-x_s)
            # directly instead of negating a gradient computed with x_s-x_f.
            grad_kernel_fluid = smoothing_kernel_grad_unsafe(neighbor_system, -pos_diff,
                                                             distance, neighbor)

            # Pair-local pressure mirroring needs p_f although the outer particle is s.
            p_b = @inbounds current_pressure(v_neighbor_system, neighbor_system, neighbor)
            p_a = @inbounds neighbor_pressure(v_particle_system, particle_system, particle,
                                              p_b)

            dv_fluid = @inbounds physical_fluid_pair_acceleration(neighbor_system,
                                                                  particle_system,
                                                                  v_neighbor_system,
                                                                  v_particle_system,
                                                                  neighbor, particle,
                                                                  m_b, m_a, p_b, p_a, rho_b,
                                                                  rho_a,
                                                                  v_b, v_a, -pos_diff,
                                                                  distance,
                                                                  sound_speed,
                                                                  grad_kernel_fluid,
                                                                  correction)

            return -m_b * dv_fluid, drho_particle
        end

        # TLSPH converts force with material mass; rigid bodies accumulate particle forces.
        @inbounds write_fluid_force!(dv, particle_system, force_particle_, particle)
        @inbounds write_drho_particle!(dv, particle_system, drho_particle_,
                                       particle)
    end

    return dv
end

@inline function add_continuity_equation(drho_particle,
                                         particle_system::AbstractStructureSystem,
                                         neighbor_system::AbstractFluidSystem,
                                         particle, neighbor, pos_diff, distance,
                                         m_b, rho_a, rho_b, v_a, v_b)
    return drho_particle
end

@inline function add_continuity_equation(drho_particle,
                                         particle_system::Union{RigidBodySystem{<:BoundaryModelDummyParticles{ContinuityDensity}},
                                                                TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles{ContinuityDensity}}},
                                         neighbor_system::AbstractFluidSystem,
                                         particle, neighbor, pos_diff, distance,
                                         m_b, rho_a, rho_b, v_a, v_b)
    # Continuity uses (v_s-v_f) dot grad_s W from the hydrodynamic boundary kernel.
    # The fluid formulation weights it by m_f (summation) or (rho_s/rho_f)*m_f (continuity).
    grad_kernel = hydrodynamic_kernel_grad(particle_system, pos_diff, distance, particle)

    return add_continuity_equation(drho_particle,
                                   density_calculator(neighbor_system),
                                   m_b, rho_a, rho_b, v_a, v_b, grad_kernel, particle)
end

@inline function write_drho_particle!(dv, ::AbstractSystem, drho_particle, particle)
    return dv
end

@propagate_inbounds function write_drho_particle!(dv,
                                                  ::Union{RigidBodySystem{<:BoundaryModelDummyParticles{ContinuityDensity}},
                                                          TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles{ContinuityDensity}}},
                                                  drho_particle, particle)
    dv[end, particle] += drho_particle

    return dv
end

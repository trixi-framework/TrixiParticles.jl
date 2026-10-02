# Shared structure-fluid interaction helpers used by multiple structure schemes.
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

    # Use the same pair cutoffs as the fluid RHS.
    compact_support_ = compact_support(neighbor_system, particle_system)
    almostzero = interaction_zero_distance(neighbor_system, particle_system)

    @threaded semi for particle in eachparticle
        # In fluid-structure interaction, use the "hydrodynamic mass" of the structure particles
        # corresponding to the rest density of the fluid and not the material density.
        m_a = @inbounds hydrodynamic_mass(particle_system, particle)
        rho_a = @inbounds current_density(v_particle_system, particle_system, particle)
        v_a = @inbounds current_velocity(v_particle_system, particle_system, particle)

        # Accumulate force and density rate separately, as in the fluid RHS.
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

            skip_fluid_pair(neighbor_system, distance, compact_support_, almostzero) &&
                return init

            drho_particle = @inbounds add_continuity_equation(zero(rho_a),
                                                              particle_system,
                                                              neighbor_system,
                                                              particle, neighbor, pos_diff,
                                                              distance, m_b, rho_a, rho_b,
                                                              v_a, v_b)

            # Corrected gradients need not be odd; evaluate the fluid-side gradient.
            grad_kernel_fluid = smoothing_kernel_grad_unsafe(neighbor_system, -pos_diff,
                                                             distance, neighbor)

            # Use the hydrodynamic pressure defined by the structure's boundary model.
            p_b = @inbounds current_pressure(v_neighbor_system, neighbor_system, neighbor)
            p_a = @inbounds neighbor_pressure(v_particle_system, particle_system, particle,
                                              p_b)

            # Fluid-first ordering preserves the actual fluid force, including the
            # approaching-particle condition of artificial viscosity.
            dv_fluid = @inbounds fluid_pair_acceleration(neighbor_system, particle_system,
                                                         v_neighbor_system,
                                                         v_particle_system,
                                                         neighbor, particle,
                                                         m_b, m_a, p_b, p_a, rho_b, rho_a,
                                                         v_b, v_a, -pos_diff, distance,
                                                         sound_speed, grad_kernel_fluid,
                                                         correction)

            return -m_b * dv_fluid, drho_particle
        end

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
    # Boundary density retains its existing structure-first gradient.
    grad_kernel = smoothing_kernel_grad_unsafe(neighbor_system, pos_diff, distance,
                                               neighbor)

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

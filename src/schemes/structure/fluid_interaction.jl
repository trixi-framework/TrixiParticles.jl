# Structure-fluid coupling shared by TLSPH and rigid-body systems.
@inline average_pressure(system, particle) = zero(eltype(system))

@inline interaction_force_correction(system, rho_a, rho_b) = (1, 1, 1)

@inline function interaction_force_correction(system::WeaklyCompressibleSPHSystem,
                                              rho_a, rho_b)
    return free_surface_correction(system_correction(system), system, rho_a, rho_b)
end

@inline function interaction_zero_distance(system, neighbor_system)
    return sqrt(eps(initial_smoothing_length(system)^2))
end

@inline function interaction_zero_distance(system::WeaklyCompressibleSPHSystem,
                                           neighbor_system)
    return sqrt(eps(compact_support(system, neighbor_system)^2))
end

@inline function skip_fluid_pair(system, distance, support, almostzero)
    return distance > support || (skip_zero_distance(system) && distance < almostzero)
end

@inline function sum_interaction_contributions(a, b)
    dv_a, drho_a = a
    dv_b, drho_b = b
    return dv_a + dv_b, drho_a + drho_b
end

# Evaluate the fluid particle's physical acceleration to assemble its structural
# reaction. Shifting transport terms are excluded from the interfacial traction.
@propagate_inbounds function physical_fluid_pair_acceleration(particle_system,
                                                              neighbor_system,
                                                              v_particle_system,
                                                              v_neighbor_system,
                                                              particle, neighbor,
                                                              m_a, m_b, p_a, p_b, rho_a,
                                                              rho_b,
                                                              v_a, v_b, pos_diff, distance,
                                                              sound_speed, grad_kernel,
                                                              correction)
    p_avg = average_pressure(particle_system, particle)
    viscosity_correction, pressure_correction,
    surface_tension_correction = interaction_force_correction(particle_system, rho_a, rho_b)
    dv_pressure = pressure_acceleration(particle_system, neighbor_system,
                                        particle, neighbor, m_a, m_b,
                                        p_a - p_avg, p_b - p_avg, rho_a, rho_b,
                                        pos_diff, distance, grad_kernel, correction)
    dv_particle = dv_pressure * pressure_correction
    dv_particle = add_dv_viscosity(dv_particle, particle_system, neighbor_system,
                                   v_particle_system, v_neighbor_system,
                                   particle, neighbor, pos_diff, distance, sound_speed,
                                   m_a, m_b, rho_a, rho_b, v_a, v_b, grad_kernel,
                                   viscosity_correction)
    surface_tension_a = surface_tension_model(particle_system)
    surface_tension_b = surface_tension_model(neighbor_system)
    dv_particle = add_dv_surface_tension(dv_particle, surface_tension_a, surface_tension_b,
                                         particle_system, neighbor_system, particle,
                                         neighbor,
                                         pos_diff, distance, rho_a, rho_b, grad_kernel,
                                         surface_tension_correction)
    return add_dv_adhesion(dv_particle, surface_tension_a,
                           particle_system, neighbor_system, particle, neighbor,
                           pos_diff, distance)
end

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

            # Boundary density has its own support and near-zero check.
            drho_particle = @inbounds add_continuity_equation(zero(rho_a),
                                                              particle_system,
                                                              neighbor_system,
                                                              particle, neighbor, pos_diff,
                                                              distance, m_b, rho_a, rho_b,
                                                              v_a, v_b)

            # Apply the fluid-side cutoffs only to physical momentum contributions.
            if skip_fluid_pair(neighbor_system, distance, compact_support_, almostzero)
                return zero(v_a), drho_particle
            end

            # Corrected gradients need not be odd; evaluate the fluid-side gradient.
            grad_kernel_fluid = smoothing_kernel_grad_unsafe(neighbor_system, -pos_diff,
                                                             distance, neighbor)

            # Use the hydrodynamic pressure defined by the structure's boundary model.
            p_b = @inbounds current_pressure(v_neighbor_system, neighbor_system, neighbor)
            p_a = @inbounds neighbor_pressure(v_particle_system, particle_system, particle,
                                              p_b)

            # Fluid-first ordering preserves the actual fluid force, including the
            # approaching-particle condition of artificial viscosity.
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
    # Hydrodynamic density uses the boundary kernel, not the neighboring fluid's
    # correction or a TLSPH elastic self-interaction kernel.
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

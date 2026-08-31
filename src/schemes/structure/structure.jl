# Shared structure-fluid interaction helpers used by multiple structure schemes.
@propagate_inbounds function accumulate_structure_fluid_pair!(dv, dv_fs,
                                                              particle_system::TotalLagrangianSPHSystem,
                                                              particle, m_b)
    material_mass = particle_system.mass[particle]
    for dim in eachindex(dv_fs)
        dv[dim, particle] += dv_fs[dim] * m_b / material_mass
    end
end

@propagate_inbounds function accumulate_structure_fluid_pair!(dv, dv_fs,
                                                              particle_system::RigidBodySystem,
                                                              particle, m_b)
    force_per_particle = particle_system.force_per_particle
    for dim in eachindex(dv_fs)
        force_per_particle[dim, particle] += dv_fs[dim] * m_b
    end
end

function interact_structure_fluid!(dv, v_particle_system, u_particle_system,
                                   v_neighbor_system, u_neighbor_system,
                                   particle_system,
                                   neighbor_system::AbstractFluidSystem, semi;
                                   eachparticle=each_integrated_particle(particle_system))
    sound_speed = system_sound_speed(neighbor_system)
    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # For `distance == 0`, the analytical gradient is zero, but the unsafe gradient
    # and the density diffusion divide by zero.
    # To account for rounding errors, we check if `distance` is almost zero.
    # Since the coordinates are in the order of the smoothing length `h`, `distance^2` is in
    # the order of `h^2`, so we need to check `distance < sqrt(eps(h^2))`.
    # Note that `sqrt(eps(h^2)) != eps(h)`.
    h = initial_smoothing_length(neighbor_system)
    almostzero = sqrt(eps(h^2))

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    foreach_point_neighbor(particle_system, neighbor_system,
                           system_coords, neighbor_coords, semi;
                           points=eachparticle) do particle, neighbor, pos_diff, distance
        # Skip neighbors with the same position because the kernel gradient is zero.
        # Note that `return` only exits the closure, i.e., skips the current neighbor.
        skip_zero_distance(neighbor_system) && distance < almostzero && return

        # The structure-oriented gradient is used by viscosity and adhesion below.
        grad_kernel = smoothing_kernel_grad_unsafe(neighbor_system, pos_diff,
                                                   distance, neighbor)

        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        rho_a = current_density(v_particle_system, particle_system, particle)
        rho_b = current_density(v_neighbor_system, neighbor_system, neighbor)

        v_a = current_velocity(v_particle_system, particle_system, particle)
        v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)

        surface_tension = surface_tension_model(neighbor_system)

        # In fluid-structure interaction, use the "hydrodynamic mass" of the structure particles
        # corresponding to the rest density of the fluid and not the material density.
        m_a = hydrodynamic_mass(particle_system, particle)

        # In fluid-structure interaction, use the "hydrodynamic pressure" of the structure
        # particles corresponding to the chosen boundary model.
        p_fluid = current_pressure(v_neighbor_system, neighbor_system, neighbor)
        p_boundary = neighbor_pressure(v_particle_system, particle_system, particle,
                                       p_fluid)
        p_avg = pair_pressure_offset(neighbor_system, particle_system, neighbor, particle)

        # Reconstruct the fluid-oriented pair exactly as in the fluid-structure interaction.
        # Corrected gradients are generally not odd, so evaluating the fluid gradient at the
        # reversed displacement would not yield the reaction force. Instead, compute the fluid
        # acceleration with the same orientation and apply its exact negative to the structure.
        fluid_pos_diff = -pos_diff
        fluid_grad_kernel = smoothing_kernel_grad_unsafe(neighbor_system, fluid_pos_diff,
                                                         distance, neighbor)
        dv_fluid_pressure = pressure_acceleration(neighbor_system, particle_system,
                                                  neighbor, particle,
                                                  m_b, m_a, p_fluid - p_avg,
                                                  p_boundary - p_avg, rho_b, rho_a,
                                                  fluid_pos_diff, distance,
                                                  fluid_grad_kernel,
                                                  system_correction(neighbor_system))
        pressure_correction = interaction_pressure_correction(neighbor_system, rho_b,
                                                              rho_a)

        dv_particle = add_dv_viscosity(-dv_fluid_pressure * pressure_correction,
                                       neighbor_system, particle_system,
                                       v_neighbor_system, v_particle_system,
                                       neighbor, particle, pos_diff, distance,
                                       sound_speed, m_b, m_a, rho_b, rho_a,
                                       v_b, v_a, grad_kernel)

        dv_particle = add_dv_adhesion(dv_particle, surface_tension,
                                      neighbor_system, particle_system,
                                      neighbor, particle, pos_diff, distance)

        accumulate_structure_fluid_pair!(dv, dv_particle, particle_system, particle, m_b)

        drho_particle = add_continuity_equation(zero(rho_a),
                                                particle_system, neighbor_system,
                                                particle, neighbor, pos_diff, distance,
                                                m_b, rho_a, rho_b, v_a, v_b, grad_kernel)

        @inbounds write_drho_particle!(dv, particle_system, drho_particle, particle)
    end

    return dv
end

@inline function add_continuity_equation(drho_particle,
                                         particle_system::AbstractStructureSystem,
                                         neighbor_system::AbstractFluidSystem,
                                         particle, neighbor, pos_diff, distance,
                                         m_b, rho_a, rho_b, v_a, v_b, grad_kernel)
    return drho_particle
end

@inline function add_continuity_equation(drho_particle,
                                         particle_system::Union{RigidBodySystem{<:BoundaryModelDummyParticles{ContinuityDensity}},
                                                                TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles{ContinuityDensity}}},
                                         neighbor_system::AbstractFluidSystem,
                                         particle, neighbor, pos_diff, distance,
                                         m_b, rho_a, rho_b, v_a, v_b, grad_kernel)
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

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

# The structure-fluid interaction computes the opposite of the force that the fluid
# experiences in the fluid-structure interaction. Therefore, both interactions must find
# the same pairs of particles, so the neighborhood search of the structure must use
# the compact support of the fluid.
# Note that this is only a restriction for `BoundaryModelDummyParticles`, since
# `BoundaryModelMonaghanKajtar` always uses the compact support of the fluid.
function check_compact_support_fsi(system, ::BoundaryModelDummyParticles,
                                   neighbor_system::AbstractFluidSystem)
    compact_support_structure = compact_support(system, neighbor_system)
    compact_support_fluid = compact_support(neighbor_system, system)

    if !isapprox(compact_support_structure, compact_support_fluid)
        throw(ArgumentError("the compact support of the boundary model of the " *
                            "`$(nameof(typeof(system)))` ($compact_support_structure) " *
                            "must be the same as the compact support of the fluid system " *
                            "($compact_support_fluid). Use the same smoothing kernel and " *
                            "smoothing length for the boundary model as for the fluid."))
    end

    return system
end

check_compact_support_fsi(system, boundary_model, neighbor_system) = system

function interact_structure_fluid!(dv, v_particle_system, u_particle_system,
                                   v_neighbor_system, u_neighbor_system,
                                   particle_system,
                                   neighbor_system::AbstractFluidSystem, semi;
                                   eachparticle=each_integrated_particle(particle_system))
    sound_speed = system_sound_speed(neighbor_system)
    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    h = initial_smoothing_length(neighbor_system)
    zero_distance_threshold = almostzero(h)
    zero_distance_mode = zero_distance_gradient_mode(neighbor_system, particle_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    foreach_point_neighbor(particle_system, neighbor_system,
                           system_coords, neighbor_coords, semi;
                           points=eachparticle) do particle, neighbor, pos_diff, distance
        # Skip neighbors with the same position when both endpoint gradients are zero.
        # Note that `return` only exits the closure, i.e., skips the current neighbor.
        skip_zero_distance(zero_distance_mode, distance, zero_distance_threshold) && return

        # The structure-oriented gradient is used by the continuity equation below.
        grad_kernel = local_smoothing_kernel_grad_unsafe(zero_distance_mode,
                                                         neighbor_system, pos_diff,
                                                         distance, neighbor,
                                                         zero_distance_threshold)

        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        rho_a = current_density(v_particle_system, particle_system, particle)
        rho_b = current_density(v_neighbor_system, neighbor_system, neighbor)

        v_a = current_velocity(v_particle_system, particle_system, particle)
        v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)

        # In fluid-structure interaction, use the "hydrodynamic mass" of the structure particles
        # corresponding to the rest density of the fluid and not the material density.
        m_a = hydrodynamic_mass(particle_system, particle)

        # In fluid-structure interaction, use the "hydrodynamic pressure" of the structure
        # particles corresponding to the chosen boundary model.
        p_fluid = current_pressure(v_neighbor_system, neighbor_system, neighbor)
        p_boundary = neighbor_pressure(v_particle_system, particle_system, particle,
                                       p_fluid)

        # Reconstruct the fluid-oriented pair exactly as in the fluid-structure interaction.
        # Corrected gradients are generally not odd, so evaluating the fluid gradient at the
        # reversed displacement would not yield the reaction force. Instead, compute the fluid
        # acceleration with the same orientation and apply its exact negative to the structure.
        fluid_pos_diff = -pos_diff
        fluid_grad_kernel = local_smoothing_kernel_grad_unsafe(zero_distance_mode,
                                                               neighbor_system,
                                                               fluid_pos_diff,
                                                               distance, neighbor,
                                                               zero_distance_threshold)

        # Note that the extra terms of shifting techniques in the momentum equation are
        # intentionally not applied to the structure.
        # Shifting makes the fluid particles quasi-Lagrangian, i.e., they don't move
        # exactly with the fluid velocity. The extra terms correct for this by accounting
        # for the momentum transported between fluid particles. They are not a force.
        dv_fluid = add_momentum_equation(zero(v_b), neighbor_system, particle_system,
                                         v_neighbor_system, v_particle_system,
                                         neighbor, particle, fluid_pos_diff, distance,
                                         fluid_grad_kernel, sound_speed, m_b, m_a,
                                         p_fluid, p_boundary,
                                         rho_b, rho_a, v_b, v_a)
        dv_particle = -dv_fluid

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

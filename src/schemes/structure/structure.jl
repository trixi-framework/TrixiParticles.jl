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

@inline function pressure_acceleration_interparticle(particle_system,
                                                     neighbor_system::Union{TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles},
                                                                            RigidBodySystem{<:BoundaryModelDummyParticles}},
                                                     particle, neighbor, m_a, m_b, p_a, p_b,
                                                     rho_a, rho_b, pos_diff, distance, W_a,
                                                     correction::Union{KernelCorrection,
                                                                       GradientCorrection,
                                                                       BlendedGradientCorrection,
                                                                       MixedKernelGradientCorrection})
    # Use the boundary model's kernel and correction data for fluid pressure and TVF.
    return pressure_acceleration_interparticle(particle_system,
                                               neighbor_system.boundary_model,
                                               particle, neighbor, m_a, m_b, p_a, p_b,
                                               rho_a, rho_b, pos_diff, distance, W_a,
                                               correction)
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
    correction = system_correction(neighbor_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    foreach_point_neighbor(particle_system, neighbor_system,
                           system_coords, neighbor_coords, semi;
                           points=eachparticle) do particle, neighbor, pos_diff, distance
        # Skip neighbors with (almost) the same position because the kernel gradient
        # is zero, but computing it would divide by zero (see `almostzero`).
        # Note that `return` only exits the closure, i.e., skips the current neighbor.
        skip_zero_distance(neighbor_system) && distance < almostzero(h) && return

        # Now that we know that `distance` is not zero, we can safely call the unsafe
        # version of the kernel gradient to avoid redundant zero checks.
        # Note that we use the `neighbor_system` to compute the kernel gradient
        # to obtain the same force as in the fluid-structure interaction.
        grad_kernel = smoothing_kernel_grad_unsafe(neighbor_system, pos_diff,
                                                   distance, neighbor)
        grad_kernel_fluid = fluid_reaction_kernel_grad(correction, neighbor_system,
                                                       pos_diff, distance, neighbor,
                                                       grad_kernel)

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
        # The following call is equivalent to
        #     `p_a = current_pressure(v_particle_system, particle_system, particle)`
        # For structures using `PressureMirroring`, this returns `p_a = p_b`, which is
        # the pressure of the fluid particle, mirroring the fluid-structure interaction.
        p_b = current_pressure(v_neighbor_system, neighbor_system, neighbor)
        p_a = neighbor_pressure(v_particle_system, particle_system, particle, p_b)

        # Compute the acceleration of the fluid particle due to the structure particle
        # with the exact same function as in the fluid-structure interaction.
        # Particle and neighbor (and the corresponding systems and particle quantities)
        # are switched, so we use the fluid-first displacement and gradient.
        # By Newton's third law, the structure particle experiences the opposite force.
        #
        # Note that the extra terms of shifting techniques in the momentum equation are
        # intentionally not applied to the structure.
        # Shifting makes the fluid particles quasi-Lagrangian, i.e., they don't move
        # exactly with the fluid velocity. The extra terms correct for this by accounting
        # for the momentum transported between fluid particles. They are not a force.
        dv_fluid = add_momentum_equation(zero(v_b), neighbor_system, particle_system,
                                         v_neighbor_system, v_particle_system,
                                         neighbor, particle, -pos_diff, distance,
                                         grad_kernel_fluid, sound_speed, m_b, m_a, p_b, p_a,
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

@inline function fluid_reaction_kernel_grad(::Nothing, system, pos_diff, distance, particle,
                                            grad_kernel)
    # Reversing the displacement changes only the sign of the uncorrected gradient.
    # Reuse it with the opposite sign to obtain the fluid-first gradient.
    return -grad_kernel
end

@inline function fluid_reaction_kernel_grad(correction, system, pos_diff, distance,
                                            particle,
                                            grad_kernel)
    # With corrections, reversing the displacement may not be equivalent to negating
    # the gradient. Evaluate it with the fluid-first displacement instead.
    return smoothing_kernel_grad_unsafe(system, -pos_diff, distance, particle)
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

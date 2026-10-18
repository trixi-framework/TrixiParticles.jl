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
# the same pairs of particles. Reusing the structure-side gradient with the opposite
# sign also requires the same smoothing kernel and smoothing length.
# Note that this is only a restriction for `BoundaryModelDummyParticles`, since
# `BoundaryModelMonaghanKajtar` always uses the compact support of the fluid.
function check_smoothing_kernel_fsi(system, boundary_model::BoundaryModelDummyParticles,
                                    neighbor_system::AbstractFluidSystem)
    smoothing_kernel_structure = boundary_model.smoothing_kernel
    smoothing_kernel_fluid = system_smoothing_kernel(neighbor_system)
    smoothing_length_structure = initial_smoothing_length(boundary_model)
    smoothing_length_fluid = initial_smoothing_length(neighbor_system)

    if smoothing_kernel_structure != smoothing_kernel_fluid ||
       smoothing_length_structure != smoothing_length_fluid
        throw(ArgumentError("the smoothing kernel and smoothing length of the boundary " *
                            "model of the `$(nameof(typeof(system)))` must be the same " *
                            "as those of the fluid system `$(nameof(typeof(neighbor_system)))`."))
    end

    return system
end

check_smoothing_kernel_fsi(system, boundary_model, neighbor_system) = system

function interact_structure_fluid!(dv, v_particle_system, u_particle_system,
                                   v_neighbor_system, u_neighbor_system,
                                   particle_system,
                                   neighbor_system::AbstractFluidSystem, semi;
                                   eachparticle=each_integrated_particle(particle_system))
    sound_speed = system_sound_speed(neighbor_system)
    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    h = initial_smoothing_length(neighbor_system)
    h_structure = initial_smoothing_length(particle_system)
    support_structure = compact_support(system_smoothing_kernel(particle_system),
                                        h_structure)
    correction = system_correction(neighbor_system)
    # TLSPH may use a different kernel for solid mechanics. Only reuse its gradient
    # when it is uncorrected and uses the same kernel and smoothing length as the fluid.
    reuse_grad_kernel = system_smoothing_kernel(particle_system) ==
                        system_smoothing_kernel(neighbor_system) &&
                        h_structure == h && system_correction(particle_system) === nothing

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    foreach_point_neighbor(particle_system, neighbor_system,
                           system_coords, neighbor_coords, semi;
                           points=eachparticle) do particle, neighbor, pos_diff, distance
        # Skip neighbors with (almost) the same position because the kernel gradient
        # is zero, but computing it would divide by zero (see `almostzero`).
        # Note that `return` only exits the closure, i.e., skips the current neighbor.
        skip_zero_distance(neighbor_system) && distance < almostzero(h) && return

        # The continuity equation uses the structure-side gradient. It may be zero
        # at coincidence even when the corrected fluid gradient is nonzero.
        grad_kernel = if distance >= support_structure ||
                         (skip_zero_distance(particle_system) &&
                          distance < almostzero(h_structure))
            zero(pos_diff)
        else
            smoothing_kernel_grad_unsafe(particle_system, pos_diff, distance, particle)
        end
        grad_kernel_fluid = if reuse_grad_kernel
            fluid_kernel_grad_unsafe(correction, neighbor_system,
                                     pos_diff, distance, neighbor, grad_kernel)
        else
            smoothing_kernel_grad_unsafe(neighbor_system, -pos_diff, distance, neighbor)
        end

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
        p_a = current_pressure(v_particle_system, particle_system, particle)
        p_b = current_pressure(v_neighbor_system, neighbor_system, neighbor)

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

@inline function fluid_kernel_grad_unsafe(correction, system, pos_diff, distance, particle,
                                          grad_kernel)
    # Matching uncorrected kernels change sign with the displacement.
    # Reuse it with the opposite sign to obtain the fluid-first gradient.
    return -grad_kernel
end

@inline function fluid_kernel_grad_unsafe(::Union{KernelCorrection, GradientCorrection,
                                                  BlendedGradientCorrection,
                                                  MixedKernelGradientCorrection},
                                          system, pos_diff, distance, particle,
                                          grad_kernel)
    # These corrections depend on the fluid particle's local neighborhood, so its
    # gradient cannot be obtained by negating the structure-side gradient.
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

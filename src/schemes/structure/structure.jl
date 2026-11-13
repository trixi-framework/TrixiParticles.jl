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

# The interaction of a fluid particle with a structure particle evaluates the kernel
# gradient of the fluid and the kernel gradient of the boundary model of the structure
# (in the asymmetric pressure acceleration and in the continuity equation of the boundary
# particles). The gradient of a corrected kernel vanishes at zero distance for no correction,
# `ShepardKernelCorrection`, `AkinciFreeSurfaceCorrection`, `GradientCorrection` and
# `BlendedGradientCorrection`, but not for `KernelCorrection` and
# `MixedKernelGradientCorrection` (see `skip_zero_distance(correction)`).
# Particles at the same position can therefore only be skipped if both gradients vanish.
@inline function skip_zero_distance(particle_system::AbstractFluidSystem,
                                    neighbor_system::Union{TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles},
                                                           RigidBodySystem{<:BoundaryModelDummyParticles}})
    return skip_zero_distance(particle_system) &&
           skip_zero_distance(correction_gradient(neighbor_system.boundary_model.correction))
end

# Same as above, but with the kernel gradient `W_b` at the structure particle already
# computed. The structure-fluid interaction computes it once for the pressure acceleration
# and the continuity equation of the boundary particles.
@inline function pressure_acceleration(particle_system,
                                       neighbor_system::Union{TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles},
                                                              RigidBodySystem{<:BoundaryModelDummyParticles}},
                                       particle, neighbor, m_a, m_b, p_a, p_b, rho_a, rho_b,
                                       pos_diff, distance, W_a,
                                       correction,
                                       W_b::AbstractVector)
    # With correction, the kernel gradient is not necessarily symmetric, so call the
    # asymmetric version of the pressure acceleration formulation.
    return pressure_acceleration_formulation(particle_system)(m_a, m_b, rho_a, rho_b,
                                                              p_a, p_b, W_a, W_b)
end

# Kernel gradient at the structure particle that is required by the pressure acceleration
# of a fluid with an asymmetric corrected kernel gradient (see above).
# It is `nothing` when the pressure acceleration doesn't require it.
@inline function boundary_pressure_kernel_grad(fluid_correction, particle_system,
                                               pos_diff, distance, particle)
    return nothing
end

@inline function boundary_pressure_kernel_grad(fluid_correction,
                                               particle_system::Union{TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles},
                                                                      RigidBodySystem{<:BoundaryModelDummyParticles}},
                                               pos_diff, distance, particle)
    if has_asymmetric_gradient(fluid_correction) ||
       has_asymmetric_gradient(hydrodynamic_correction(particle_system))
        return smoothing_kernel_grad(particle_system.boundary_model, pos_diff, distance,
                                     particle)
    end
    return nothing
end

# Both FSI directions must find the same particle pairs and use matching boundary and
# fluid kernels. The TLSPH kernel for solid mechanics may differ.
# Note that this is only a restriction for `BoundaryModelDummyParticles`, since
# `BoundaryModelMonaghanKajtar` always uses the compact support of the fluid.
function check_smoothing_kernel_fsi(system, boundary_model::BoundaryModelDummyParticles,
                                    neighbor_system::AbstractFluidSystem)
    smoothing_kernel_structure = boundary_model.smoothing_kernel
    smoothing_kernel_fluid = system_smoothing_kernel(neighbor_system)
    smoothing_length_structure = initial_smoothing_length(boundary_model)
    smoothing_length_fluid = initial_smoothing_length(neighbor_system)

    # Allow rounding differences, e.g., from computing the smoothing lengths separately
    if smoothing_kernel_structure != smoothing_kernel_fluid ||
       !isapprox(smoothing_length_structure, smoothing_length_fluid)
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
    zero_distance_threshold = almostzero(h)
    zero_distance_mode = zero_distance_gradient_mode(neighbor_system, particle_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    foreach_point_neighbor(particle_system, neighbor_system,
                           system_coords, neighbor_coords, semi;
                           points=eachparticle) do particle, neighbor, pos_diff, distance
        # Skip neighbors with the same position when both endpoint gradients are zero.
        # Note that `return` only exits the closure, i.e., skips the current neighbor.
        skip_zero_distance(zero_distance_mode, distance, zero_distance_threshold) && return

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

        # The kernel gradient of the boundary model at the structure particle is required
        # by the pressure acceleration of a fluid with an asymmetric corrected kernel
        # gradient and by the continuity equation of the boundary particles.
        # When the pressure acceleration requires it, compute it only once for both.
        # Here `pos_diff` is `x_structure - x_fluid`, matching the displacement
        # passed to the boundary model in the fluid-structure interaction.
        grad_kernel_boundary = boundary_pressure_kernel_grad(system_correction(neighbor_system),
                                                             particle_system, pos_diff,
                                                             distance, particle)

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
                                         neighbor, particle, fluid_pos_diff, distance,
                                         fluid_grad_kernel, sound_speed, m_b, m_a,
                                         p_fluid, p_boundary,
                                         rho_b, rho_a, v_b, v_a, grad_kernel_boundary)
        dv_particle = -dv_fluid

        accumulate_structure_fluid_pair!(dv, dv_particle, particle_system, particle, m_b)

        drho_particle = add_continuity_equation(zero(rho_a),
                                                particle_system, neighbor_system,
                                                particle, neighbor, pos_diff, distance,
                                                m_b, rho_a, rho_b, v_a, v_b,
                                                fluid_grad_kernel, grad_kernel_boundary)

        @inbounds write_drho_particle!(dv, particle_system, drho_particle, particle)
    end

    return dv
end

@inline function add_continuity_equation(drho_particle,
                                         particle_system::AbstractStructureSystem,
                                         neighbor_system::AbstractFluidSystem,
                                         particle, neighbor, pos_diff, distance,
                                         m_b, rho_a, rho_b, v_a, v_b, grad_kernel_fluid,
                                         grad_kernel_boundary)
    return drho_particle
end

# `grad_kernel_boundary` is the kernel gradient of the boundary model at the structure
# particle if it has already been computed for the pressure acceleration, or `nothing`.
@inline function add_continuity_equation(drho_particle,
                                         particle_system::RigidBodySystem{<:BoundaryModelDummyParticles{ContinuityDensity}},
                                         neighbor_system::AbstractFluidSystem,
                                         particle, neighbor, pos_diff, distance,
                                         m_b, rho_a, rho_b, v_a, v_b, grad_kernel_fluid,
                                         grad_kernel_boundary)
    # Reuse the boundary gradient computed for the pressure acceleration when available.
    # Otherwise compute it with the boundary model's own correction, which may differ
    # from the fluid correction.
    grad_kernel = isnothing(grad_kernel_boundary) ?
                  smoothing_kernel_grad(particle_system.boundary_model, pos_diff,
                                        distance, particle) :
                  grad_kernel_boundary

    return add_continuity_equation(drho_particle,
                                   density_calculator(neighbor_system),
                                   m_b, rho_a, rho_b, v_a, v_b, grad_kernel, particle)
end

@inline function write_drho_particle!(dv, ::AbstractSystem, drho_particle, particle)
    return dv
end

@propagate_inbounds function write_drho_particle!(dv,
                                                  ::RigidBodySystem{<:BoundaryModelDummyParticles{ContinuityDensity}},
                                                  drho_particle, particle)
    dv[end, particle] += drho_particle

    return dv
end

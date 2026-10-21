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

# The density of a structure boundary is computed by the pressure extrapolation or
# integrated with the continuity equation. The summation density is only computed for
# a `WallBoundarySystem`, so the density of a structure would stay at its initial value.
function check_boundary_density_fsi(system,
                                    boundary_model::BoundaryModelDummyParticles{SummationDensity},
                                    neighbor_system::AbstractFluidSystem)
    throw(ArgumentError("`BoundaryModelDummyParticles` with density calculator " *
                        "`SummationDensity` is not yet supported for a " *
                        "`$(nameof(typeof(system)))` in a fluid-structure interaction"))
end

check_boundary_density_fsi(system, boundary_model, neighbor_system) = system

function interact_structure_fluid!(dv, v_particle_system, u_particle_system,
                                   v_neighbor_system, u_neighbor_system,
                                   particle_system,
                                   neighbor_system::AbstractFluidSystem, semi;
                                   eachparticle=each_integrated_particle(particle_system))
    sound_speed = system_sound_speed(neighbor_system)
    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    h = initial_smoothing_length(neighbor_system)

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
        # Evaluate the gradient exactly as in the fluid-structure interaction, i.e.,
        # with the fluid particle's kernel and correction and the fluid-first displacement.
        # Note that we cannot just flip the sign of the structure-first gradient, since
        # corrections like `KernelCorrection` add a term that keeps its sign
        # when the displacement is reversed.
        grad_kernel_fluid = smoothing_kernel_grad_unsafe(neighbor_system, -pos_diff,
                                                         distance, neighbor)

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
                                                m_b, rho_a, rho_b, v_a, v_b,
                                                grad_kernel_fluid)

        @inbounds write_drho_particle!(dv, particle_system, drho_particle, particle)
    end

    return dv
end

@inline function add_continuity_equation(drho_particle,
                                         particle_system::AbstractStructureSystem,
                                         neighbor_system::AbstractFluidSystem,
                                         particle, neighbor, pos_diff, distance,
                                         m_b, rho_a, rho_b, v_a, v_b, grad_kernel_fluid)
    return drho_particle
end

@inline function add_continuity_equation(drho_particle,
                                         particle_system::Union{RigidBodySystem{<:BoundaryModelDummyParticles{ContinuityDensity}},
                                                                TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles{ContinuityDensity}}},
                                         neighbor_system::AbstractFluidSystem,
                                         particle, neighbor, pos_diff, distance,
                                         m_b, rho_a, rho_b, v_a, v_b, grad_kernel_fluid)
    (; boundary_model) = particle_system

    # As for a `WallBoundarySystem`, the density of the boundary particles evolves with
    # the boundary model's kernel gradient and not with the corrected fluid gradient.
    grad_kernel = boundary_continuity_kernel_grad(system_correction(neighbor_system),
                                                  boundary_model.correction,
                                                  boundary_model, pos_diff, distance,
                                                  particle, grad_kernel_fluid)

    return add_continuity_equation(drho_particle,
                                   density_calculator(neighbor_system),
                                   m_b, rho_a, rho_b, v_a, v_b, grad_kernel, particle)
end

@inline function boundary_continuity_kernel_grad(fluid_correction, boundary_correction,
                                                 boundary_model, pos_diff, distance,
                                                 particle, grad_kernel_fluid)
    # Use the boundary model's kernel and smoothing length, which may differ from
    # the TLSPH kernel for solid mechanics.
    # The safe version also returns zero for coincident particles.
    return smoothing_kernel_grad(boundary_model, pos_diff, distance, particle)
end

@inline function boundary_continuity_kernel_grad(::Union{Nothing,
                                                         AkinciFreeSurfaceCorrection,
                                                         ShepardKernelCorrection},
                                                 ::Union{Nothing,
                                                         AkinciFreeSurfaceCorrection,
                                                         ShepardKernelCorrection},
                                                 boundary_model, pos_diff, distance,
                                                 particle, grad_kernel_fluid)
    # Neither correction changes the gradient, and the boundary model's kernel and
    # smoothing length match the fluid's (see `check_smoothing_kernel_fsi`).
    # Reversing the displacement only flips the sign of the uncorrected gradient.
    return -grad_kernel_fluid
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

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

@propagate_inbounds function accumulate_structure_fluid_pair!(dv, dv_fs,
                                                              particle_system::WallBoundarySystem,
                                                              particle, m_b)
    reaction_force = particle_system.cache.reaction_force
    for dim in eachindex(dv_fs)
        reaction_force[dim, particle] += dv_fs[dim] * m_b
    end
end

function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::WallBoundarySystem{BM, ELTYPE, NDIMS, IC, CO, M, IM,
                                                       CA},
                   neighbor_system::AbstractFluidSystem,
                   semi) where {DC <: AdamiPressureExtrapolation,
                                BM <: BoundaryModelDummyParticles{DC}, ELTYPE, NDIMS, IC,
                                CO, M <: BoundaryAttachment, IM, CA}
    return interact_structure_fluid!(dv, v_particle_system, u_particle_system,
                                     v_neighbor_system, u_neighbor_system,
                                     particle_system, neighbor_system, semi;
                                     eachparticle=eachparticle(particle_system))
end

@inline function add_continuity_equation(drho_particle,
                                         particle_system::WallBoundarySystem,
                                         neighbor_system::AbstractFluidSystem,
                                         particle, neighbor, pos_diff, distance,
                                         m_b, rho_a, rho_b, v_a, v_b, grad_kernel)
    return drho_particle
end

@inline function attachment_parent(attachment::BoundaryAttachment, semi)
    return semi.systems[parent_system_index(attachment)]
end

function update_boundary_positions!(system, attachment::BoundaryAttachment,
                                    v_ode, u_ode, semi, t)
    parent = attachment_parent(attachment, semi)
    v_parent = wrap_v(v_ode, parent, semi)
    (; parent_particles, parent_weights) = attachment
    (; coordinates, cache) = system
    (; velocity, acceleration) = cache

    @threaded semi for ghost in eachparticle(system)
        ghost_coordinates = zero(extract_svector(coordinates, system, ghost))
        ghost_velocity = zero(extract_svector(velocity, system, ghost))
        ghost_acceleration = zero(extract_svector(acceleration, system, ghost))

        for support in axes(parent_particles, 1)
            weight = @inbounds parent_weights[support, ghost]
            iszero(weight) && continue
            parent_particle = @inbounds parent_particles[support, ghost]

            ghost_coordinates += weight * current_coords(parent, parent_particle)
            ghost_velocity += weight * current_velocity(v_parent, parent, parent_particle)
            ghost_acceleration += weight * current_acceleration(parent, parent_particle)
        end

        for dim in 1:ndims(system)
            @inbounds coordinates[dim, ghost] = ghost_coordinates[dim]
            @inbounds velocity[dim, ghost] = ghost_velocity[dim]
            @inbounds acceleration[dim, ghost] = ghost_acceleration[dim]
        end
    end

    return system
end

function update_boundary_normals!(system, attachment::BoundaryAttachment, semi)
    parent = attachment_parent(attachment, semi)
    (; parent_particles, parent_weights) = attachment
    reference_normals = system.initial_condition.normals
    current_normals = system.cache.normals

    @threaded semi for ghost in eachparticle(system)
        deformation_gradient_ = zero(deformation_gradient(parent, 1))
        for support in axes(parent_particles, 1)
            weight = @inbounds parent_weights[support, ghost]
            iszero(weight) && continue
            parent_particle = @inbounds parent_particles[support, ghost]
            deformation_gradient_ += weight * deformation_gradient(parent, parent_particle)
        end

        reference_normal = extract_svector(reference_normals, system, ghost)
        determinant = det(deformation_gradient_)
        valid_deformation = isfinite(determinant) &&
                            abs(determinant) > sqrt(eps(one(determinant)))
        current_normal = valid_deformation ?
                         inv(deformation_gradient_)' * reference_normal : reference_normal
        current_norm2 = dot(current_normal, current_normal)
        if !(isfinite(current_norm2) && current_norm2 > eps(current_norm2))
            current_normal = reference_normal
            current_norm2 = dot(current_normal, current_normal)
        end
        current_normal /= sqrt(current_norm2)

        for dim in 1:ndims(system)
            @inbounds current_normals[dim, ghost] = current_normal[dim]
        end
    end

    return system
end

function finalize_boundary_interaction!(system, attachment::BoundaryAttachment,
                                        dv_ode, v_ode, u_ode, semi)
    parent = attachment_parent(attachment, semi)
    dv_parent = wrap_v(dv_ode, parent, semi)
    reaction_force = system.cache.reaction_force
    (; ghost_particles_by_parent, ghost_weights_by_parent) = attachment

    @threaded semi for parent_particle in each_integrated_particle(parent)
        force = zero(extract_svector(dv_parent, parent, parent_particle))
        for slot in axes(ghost_particles_by_parent, 1)
            ghost = @inbounds ghost_particles_by_parent[slot, parent_particle]
            iszero(ghost) && continue
            weight = @inbounds ghost_weights_by_parent[slot, parent_particle]
            force += weight * extract_svector(reaction_force, system, ghost)
        end

        for dim in 1:ndims(parent)
            @inbounds dv_parent[dim,
                                parent_particle] += force[dim] /
                                                    parent.mass[parent_particle]
        end
    end

    update_fsi_acceleration!(parent, dv_parent, semi)
    return system
end

function check_boundary_attachment(system, attachment::BoundaryAttachment, systems)
    parent_index = parent_system_index(attachment)
    parent_index <= length(systems) ||
        throw(ArgumentError("attached boundary parent system index $parent_index is out of bounds"))
    parent = systems[parent_index]
    parent isa TotalLagrangianSPHSystem ||
        throw(ArgumentError("an attached boundary parent must be a `TotalLagrangianSPHSystem`"))
    ndims(parent) == ndims(system) ||
        throw(ArgumentError("an attached boundary and its parent must have the same number of dimensions"))

    system_index = findfirst(candidate -> candidate === system, systems)
    parent_index < system_index ||
        throw(ArgumentError("the parent `TotalLagrangianSPHSystem` must precede its attached boundary"))
    system.boundary_model isa BoundaryModelDummyParticles{<:AdamiPressureExtrapolation} ||
        throw(ArgumentError("an attached boundary currently requires `AdamiPressureExtrapolation`"))
    any(parent.hydrodynamic_boundary) &&
        throw(ArgumentError("the attached parent must use `hydrodynamic_boundary_particles=Int[]` " *
                            "to avoid duplicate fluid coupling"))

    size(attachment.parent_particles, 2) == nparticles(system) ||
        throw(ArgumentError("the attachment map must have one column per boundary particle"))
    size(attachment.ghost_particles_by_parent, 2) == nparticles(parent) ||
        throw(ArgumentError("the attachment reverse map must have one column per parent particle"))

    for ghost in eachparticle(system)
        reference_normal = extract_svector(system.initial_condition.normals, system, ghost)
        normal_norm2 = dot(reference_normal, reference_normal)
        isfinite(normal_norm2) && normal_norm2 > eps(normal_norm2) ||
            throw(ArgumentError("attached boundary particle $ghost requires a finite, nonzero reference normal"))

        for dim in 1:ndims(system)
            mapped_coordinate = zero(eltype(system))
            for support in axes(attachment.parent_particles, 1)
                weight = attachment.parent_weights[support, ghost]
                iszero(weight) && continue
                parent_particle = attachment.parent_particles[support, ghost]
                mapped_coordinate += weight *
                                     parent.initial_coordinates[dim, parent_particle]
            end
            coordinate = system.initial_condition.coordinates[dim, ghost]
            isapprox(mapped_coordinate, coordinate; rtol=sqrt(eps(eltype(system))),
                     atol=sqrt(eps(eltype(system)))) ||
                throw(ArgumentError("attached boundary particle $ghost does not match its parent map"))
        end
    end

    return system
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
    zero_distance_mode = zero_distance_gradient_mode(neighbor_system, particle_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    foreach_point_neighbor(particle_system, neighbor_system,
                           system_coords, neighbor_coords, semi;
                           points=eachparticle) do particle, neighbor, pos_diff, distance
        is_hydrodynamic_particle(particle_system, particle) || return

        # Skip neighbors with the same position when both endpoint gradients are zero.
        # Note that `return` only exits the closure, i.e., skips the current neighbor.
        skip_zero_distance(zero_distance_mode, distance, almostzero) && return

        # The structure-oriented gradient is used by viscosity and adhesion below.
        grad_kernel = local_smoothing_kernel_grad_unsafe(zero_distance_mode,
                                                         neighbor_system, pos_diff,
                                                         distance, neighbor, almostzero)

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
        fluid_pos_diff = -pos_diff
        p_boundary,
        v_boundary_state = apply_wall_boundary_state(p_boundary, v_a,
                                                     system_boundary_model(particle_system),
                                                     neighbor_system,
                                                     particle_system, neighbor, particle,
                                                     p_fluid, rho_b, v_b,
                                                     fluid_pos_diff, distance, sound_speed)
        p_avg = pair_pressure_offset(neighbor_system, particle_system, neighbor, particle)

        # Reconstruct the fluid-oriented pair exactly as in the fluid-structure interaction.
        # Corrected gradients are generally not odd, so evaluating the fluid gradient at the
        # reversed displacement would not yield the reaction force. Instead, compute the fluid
        # acceleration with the same orientation and apply its exact negative to the structure.
        fluid_grad_kernel = local_smoothing_kernel_grad_unsafe(zero_distance_mode,
                                                               neighbor_system,
                                                               fluid_pos_diff,
                                                               distance, neighbor,
                                                               almostzero)
        dv_fluid_pressure = pressure_acceleration(neighbor_system, particle_system,
                                                  neighbor, particle,
                                                  m_b, m_a, p_fluid - p_avg,
                                                  p_boundary - p_avg, rho_b, rho_a,
                                                  fluid_pos_diff, distance,
                                                  fluid_grad_kernel,
                                                  system_correction(neighbor_system))
        (viscosity_correction, pressure_correction,
         _) = interaction_force_corrections(neighbor_system, rho_b, rho_a)

        dv_particle = add_dv_viscosity(-dv_fluid_pressure * pressure_correction,
                                       neighbor_system, particle_system,
                                       v_neighbor_system, v_particle_system,
                                       neighbor, particle, pos_diff, distance,
                                       sound_speed, m_b, m_a, rho_b, rho_a,
                                       v_b, v_boundary_state, grad_kernel,
                                       viscosity_correction)

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

@inline function interaction_force_corrections(system, rho_a, rho_b)
    one_ = one(rho_a)
    return one_, one_, one_
end

@inline function interaction_force_corrections(system::Union{WeaklyCompressibleSPHSystem,
                                                             EntropicallyDampedSPHSystem},
                                               rho_a, rho_b)
    return free_surface_correction(correction_force(system.correction), system,
                                   rho_a, rho_b)
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

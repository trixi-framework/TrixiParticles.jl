project_resting_contact_velocity!(system, v_system, u_system, v_ode, u_ode, semi,
                                  integrator) = false

@inline function rotational_velocity_2d(angular_velocity, relative_position)
    return SVector(-angular_velocity * relative_position[2],
                   angular_velocity * relative_position[1])
end

function add_or_merge_resting_constraint!(constraints,
                                          particle, normal, wall_velocity,
                                          relative_position,
                                          penetration_effective, normal_merge_cos)
    for i in eachindex(constraints)
        existing_particle, existing_normal, _, _, existing_penetration = constraints[i]
        if existing_particle == particle && dot(existing_normal, normal) >= normal_merge_cos
            if penetration_effective > existing_penetration
                constraints[i] = (particle, normal, wall_velocity, relative_position,
                                  penetration_effective)
            end
            return constraints
        end
    end

    push!(constraints, (particle, normal, wall_velocity, relative_position,
                        penetration_effective))

    return constraints
end

function reset_projected_contact_history!(contact_tangential_displacement, contact_keys,
                                          projected_particles=nothing)
    isnothing(contact_tangential_displacement) && return contact_tangential_displacement

    for key in contact_keys
        delete!(contact_tangential_displacement, key)
    end

    if !isnothing(projected_particles)
        for key in collect(keys(contact_tangential_displacement))
            key[2] in projected_particles && delete!(contact_tangential_displacement, key)
        end
    end

    return contact_tangential_displacement
end

@inline function reset_resting_contact_counter!(system::RigidSPHSystem)
    system.cache.resting_contact_counter[] = 0

    return system
end

@inline function update_resting_contact_counter!(system::RigidSPHSystem, in_resting_contact)
    if in_resting_contact
        system.cache.resting_contact_counter[] += 1
    else
        system.cache.resting_contact_counter[] = 0
    end

    return system.cache.resting_contact_counter[]
end

@inline function resting_contact_speed_threshold(contact_model::RigidBoundaryContactModel,
                                                 dt_contact, velocity_floor, ELTYPE)
    characteristic_speed = dt_contact > eps(ELTYPE) ? contact_model.contact_distance / dt_contact :
                           zero(ELTYPE)

    return max(convert(ELTYPE, 5) * velocity_floor,
               convert(ELTYPE, 0.02) * characteristic_speed)
end

"""
    valid_resting_projection_timestep!(system, dt, dt_contact)

Validate time-step inputs for resting-contact projection and reset persistence
state when inputs are invalid.
"""
@inline function valid_resting_projection_timestep!(system::RigidSPHSystem, dt, dt_contact)
    if !isfinite(dt) || dt <= 0 || !isfinite(dt_contact) || dt_contact <= 0
        reset_resting_contact_counter!(system)
        return false
    end

    return true
end

"""
    resting_projection_triggered(dt, dt_contact, resting_counter, ELTYPE)

Trigger projection either when `dt` is already near contact-time collapse or
when a resting regime persists for multiple callback updates.
"""
@inline function resting_projection_triggered(dt, dt_contact, resting_counter, ELTYPE)
    dt_projection_trigger = convert(ELTYPE, 0.1) * dt_contact
    projection_persistence_steps = 2

    return dt <= dt_projection_trigger || resting_counter >= projection_persistence_steps
end

function should_project_resting_contact!(system::RigidSPHSystem,
                                         contact_model::RigidBoundaryContactModel,
                                         dt, dt_contact,
                                         velocity_floor,
                                         max_normal_speed,
                                         max_tangential_speed,
                                         ELTYPE)
    resting_speed_max = resting_contact_speed_threshold(contact_model, dt_contact,
                                                        velocity_floor, ELTYPE)
    in_resting_contact = max_normal_speed <= resting_speed_max &&
                         max_tangential_speed <= resting_speed_max
    resting_counter = update_resting_contact_counter!(system, in_resting_contact)
    in_resting_contact || return false

    return resting_projection_triggered(dt, dt_contact, resting_counter, ELTYPE)
end

function collect_resting_contact_constraints!(system::RigidSPHSystem{<:Any, <:Any, 2},
                                              u_system, v_ode, u_ode, semi,
                                              center_of_mass_velocity,
                                              angular_velocity,
                                              contact_model::RigidBoundaryContactModel,
                                              constraints,
                                              projection_contact_keys,
                                              projected_particles)
    ELTYPE = eltype(system)
    zero_tangential = zero(SVector{2, ELTYPE})
    normal_merge_cos = convert(ELTYPE, 0.995)
    max_normal_speed = zero(eltype(system))
    max_tangential_speed = zero(eltype(system))
    system_coords = current_coordinates(u_system, system)

    foreach_system(semi) do neighbor_system
        neighbor_system isa WallBoundarySystem || return

        reset_contact_manifold_cache!(system.cache)

        neighbor_system_index = system_indices(neighbor_system, semi)
        v_neighbor = wrap_v(v_ode, neighbor_system, semi)
        u_neighbor = wrap_u(u_ode, neighbor_system, semi)
        neighbor_coords = current_coordinates(u_neighbor, neighbor_system)

        foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords, semi;
                               points=each_integrated_particle(system),
                               parallelization_backend=SerialBackend()) do particle, neighbor,
                                                                           pos_diff, distance
            distance <= eps(eltype(system)) && return

            penetration = contact_model.contact_distance - distance
            penetration_effective = penetration - contact_model.penetration_slop
            penetration_effective <= 0 && return

            normal = pos_diff / distance
            wall_velocity = current_velocity(v_neighbor, neighbor_system, neighbor)
            contact_weight = wall_contact_pair_weight(neighbor_system, distance, neighbor,
                                                      ELTYPE)
            contact_weight <= eps(ELTYPE) && return

            manifold = find_or_add_contact_manifold!(system.cache, particle, normal,
                                                     normal_merge_cos, ELTYPE)
            accumulate_contact_manifold!(system.cache, particle, manifold, contact_weight,
                                         normal, wall_velocity, penetration_effective,
                                         zero_tangential)
        end

        for particle in each_integrated_particle(system)
            n_manifolds = system.cache.contact_manifold_count[particle]
            n_manifolds == 0 && continue

            relative_position = extract_svector(system.cache.relative_coordinates, system,
                                                particle)
            particle_velocity = center_of_mass_velocity +
                                rotational_velocity_2d(angular_velocity, relative_position)

            for manifold in 1:n_manifolds
                weight_sum = system.cache.contact_manifold_weight_sum[manifold, particle]
                weight_sum <= eps(ELTYPE) && continue

                normal = weighted_manifold_vector(system.cache.contact_manifold_normal_sum,
                                                  manifold, particle, weight_sum, Val(2),
                                                  ELTYPE)
                normal_norm = norm(normal)
                normal_norm <= eps(ELTYPE) && continue
                normal /= normal_norm

                wall_velocity = weighted_manifold_vector(system.cache.contact_manifold_wall_velocity_sum,
                                                         manifold, particle, weight_sum,
                                                         Val(2), ELTYPE)
                penetration_effective = system.cache.contact_manifold_penetration_sum[manifold,
                                                                                       particle] /
                                        weight_sum
                relative_velocity = particle_velocity - wall_velocity
                normal_velocity = dot(relative_velocity, normal)
                tangential_velocity = relative_velocity - normal_velocity * normal

                max_normal_speed = max(max_normal_speed, abs(normal_velocity))
                max_tangential_speed = max(max_tangential_speed, norm(tangential_velocity))

                contact_key = (neighbor_system_index, particle, manifold)
                push!(projection_contact_keys, contact_key)
                push!(projected_particles, particle)
                add_or_merge_resting_constraint!(constraints,
                                                 particle, normal, wall_velocity,
                                                 relative_position,
                                                 penetration_effective, normal_merge_cos)
            end
        end
    end

    return max_normal_speed, max_tangential_speed
end

function project_resting_contact_position!(system::RigidSPHSystem{<:Any, <:Any, 2},
                                           u_system, constraints,
                                           penetration_tolerance,
                                           inv_inertia)
    total_mass = system.cache.total_mass
    total_mass <= eps(eltype(system)) && return false

    inv_total_mass = inv(total_mass)

    center_of_mass_correction = zero(SVector{2, eltype(system)})
    angular_correction = zero(eltype(system))
    any_correction = false

    for _ in 1:3
        iteration_has_correction = false

        for (_, normal, _, relative_position, penetration_effective) in constraints
            penetration_effective <= penetration_tolerance && continue

            applied_correction = center_of_mass_correction +
                                 rotational_velocity_2d(angular_correction,
                                                        relative_position)
            remaining_penetration = penetration_effective -
                                    dot(applied_correction, normal)
            remaining_penetration <= penetration_tolerance && continue

            normal_lever = cross(relative_position, normal)
            effective_inverse_mass = inv_total_mass +
                                     normal_lever^2 * inv_inertia
            effective_inverse_mass <= eps(eltype(system)) && continue

            correction_impulse = (remaining_penetration - penetration_tolerance) /
                                 effective_inverse_mass
            correction_impulse <= eps(eltype(system)) && continue

            center_of_mass_correction += correction_impulse * inv_total_mass * normal
            angular_correction += correction_impulse * normal_lever * inv_inertia
            iteration_has_correction = true
        end

        any_correction |= iteration_has_correction
        iteration_has_correction || break
    end

    any_correction || return false

    system_coords = current_coordinates(u_system, system)
    for particle in each_integrated_particle(system)
        relative_position = extract_svector(system.cache.relative_coordinates, system, particle)
        position_correction = center_of_mass_correction +
                              rotational_velocity_2d(angular_correction,
                                                     relative_position)
        @inbounds begin
            system_coords[1, particle] += position_correction[1]
            system_coords[2, particle] += position_correction[2]
        end
    end

    return true
end

function project_resting_contact_velocity!(system::RigidSPHSystem{<:Any, <:Any, 2},
                                           v_system, u_system, v_ode, u_ode, semi,
                                           integrator)
    contact_model = system.boundary_contact_model
    isnothing(contact_model) && return false
    contact_model.resting_contact_projection || return false
    system.cache.boundary_contact_count[] > 0 || begin
        reset_resting_contact_counter!(system)
        return false
    end

    dt = integrator.dt
    dt_contact = contact_time_step(contact_model, system)
    valid_resting_projection_timestep!(system, dt, dt_contact) || return false

    center_of_mass_velocity = system.cache.center_of_mass_velocity[]
    angular_velocity = system.cache.angular_velocity[]
    ELTYPE = eltype(system)
    velocity_floor = max(convert(ELTYPE, 0.1) *
                         contact_model.stick_velocity_tolerance,
                         sqrt(eps(ELTYPE)))

    constraint_type = Tuple{Int, SVector{2, eltype(system)}, SVector{2, eltype(system)},
                            SVector{2, eltype(system)}, eltype(system)}
    constraints = constraint_type[]
    projection_contact_keys = Set{NTuple{3, Int}}()
    projected_particles = Set{Int}()
    max_normal_speed, max_tangential_speed = collect_resting_contact_constraints!(system,
                                                                                   u_system,
                                                                                   v_ode, u_ode,
                                                                                   semi,
                                                                                   center_of_mass_velocity,
                                                                                   angular_velocity,
                                                                                   contact_model,
                                                                                   constraints,
                                                                                   projection_contact_keys,
                                                                                   projected_particles)

    if isempty(constraints)
        reset_resting_contact_counter!(system)
        return false
    end

    should_project_resting_contact!(system,
                                    contact_model,
                                    dt, dt_contact,
                                    velocity_floor,
                                    max_normal_speed,
                                    max_tangential_speed,
                                    ELTYPE) || return false

    total_mass = system.cache.total_mass
    total_mass <= eps(eltype(system)) && return false

    inv_total_mass = inv(total_mass)
    inertia = system.cache.inertia[]
    inv_inertia = (!contact_model.torque_free && inertia > eps(ELTYPE)) ? inv(inertia) :
                  zero(ELTYPE)

    projected_center_of_mass_velocity = center_of_mass_velocity
    projected_angular_velocity = angular_velocity
    velocity_modified = false

    for _ in 1:3
        any_projection = false

        for (_, normal, wall_velocity, relative_position, _) in constraints
            particle_velocity = projected_center_of_mass_velocity +
                                rotational_velocity_2d(projected_angular_velocity,
                                                       relative_position)
            relative_velocity = particle_velocity - wall_velocity
            normal_velocity = dot(relative_velocity, normal)
            normal_velocity >= -velocity_floor && continue

            normal_lever = cross(relative_position, normal)
            effective_inverse_mass = inv_total_mass +
                                     normal_lever^2 * inv_inertia
            effective_inverse_mass <= eps(eltype(system)) && continue

            impulse = -normal_velocity / effective_inverse_mass
            impulse <= eps(eltype(system)) && continue

            projected_center_of_mass_velocity += impulse * inv_total_mass * normal
            projected_angular_velocity += impulse * normal_lever * inv_inertia
            any_projection = true
            velocity_modified = true
        end

        any_projection || break
    end

    if norm(projected_center_of_mass_velocity) <= velocity_floor
        projected_center_of_mass_velocity = zero(projected_center_of_mass_velocity)
    end

    contact_scale = max(contact_model.contact_distance, system.particle_spacing)
    angular_velocity_floor = convert(ELTYPE, 0.1) * velocity_floor /
                             max(contact_scale, sqrt(eps(ELTYPE)))
    if abs(projected_angular_velocity) <= angular_velocity_floor
        projected_angular_velocity = zero(projected_angular_velocity)
    end

    velocity_modified = velocity_modified ||
                        norm(projected_center_of_mass_velocity -
                             center_of_mass_velocity) > eps(ELTYPE) ||
                        abs(projected_angular_velocity - angular_velocity) > eps(ELTYPE)

    if velocity_modified
        system_velocity = current_velocity(v_system, system)
        for particle in each_integrated_particle(system)
            relative_position = extract_svector(system.cache.relative_coordinates, system, particle)
            particle_velocity = projected_center_of_mass_velocity +
                                rotational_velocity_2d(projected_angular_velocity,
                                                       relative_position)

            @inbounds begin
                system_velocity[1, particle] = particle_velocity[1]
                system_velocity[2, particle] = particle_velocity[2]
            end
        end
    end

    penetration_tolerance = max(convert(ELTYPE, 1.0e-3) * contact_scale,
                                sqrt(eps(ELTYPE)) * contact_scale)
    position_modified = project_resting_contact_position!(system, u_system, constraints,
                                                          penetration_tolerance,
                                                          inv_inertia)
    state_modified = velocity_modified || position_modified
    state_modified || return false

    projection_time = hasproperty(integrator, :t) ? integrator.t : zero(ELTYPE)
    update_final!(system, v_system, u_system, v_ode, u_ode, semi, projection_time)
    reset_resting_contact_counter!(system)
    reset_projected_contact_history!(system.cache.contact_tangential_displacement,
                                     projection_contact_keys,
                                     projected_particles)

    return true
end

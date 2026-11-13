@inline function requires_update_callback(contact_model::RigidContactModel)
    return has_tangential_contact(contact_model) &&
           contact_model.tangential_stiffness > 0
end

function create_cache_contact_history(contact_model::RigidContactModel, ::Val{NDIMS},
                                      ::Type{ELTYPE}) where {NDIMS, ELTYPE}
    if has_tangential_contact(contact_model)
        # These dictionaries are persistent accepted-step state. In contrast, the manifold
        # arrays in `create_cache_contact_manifold` are rebuilt during every RHS evaluation.
        contact_tangential_displacement = Dict{RigidContactKey,
                                               SVector{NDIMS, ELTYPE}}()
        contact_sliding = Dict{RigidContactKey, Bool}()
        wall_contact_descriptors = Dict{RigidContactKey,
                                        WallContactDescriptor{NDIMS, ELTYPE}}()
        next_wall_contact_id = Ref(1)
    else
        contact_tangential_displacement = nothing
        contact_sliding = nothing
        wall_contact_descriptors = nothing
        next_wall_contact_id = nothing
    end

    return (; contact_tangential_displacement, contact_sliding, wall_contact_descriptors,
            next_wall_contact_id)
end

@inline function requires_update_callback(system::RigidBodySystem)
    return !isnothing(system.contact_model) &&
           requires_update_callback(system.contact_model)
end

@inline function requires_update_callback(system::RigidBodySystem, semi)
    contact_model = system.contact_model
    isnothing(contact_model) && return false

    # A model cannot decide this in isolation: rigid-rigid contact uses pair parameters and
    # a normal-only neighbor can disable friction through the minimum-coefficient rule.
    for neighbor_system in semi.systems
        neighbor_system === system && continue
        has_system_interaction(system, neighbor_system, semi) || continue

        if neighbor_system isa WallBoundarySystem
            requires_update_callback(contact_model) && return true
        elseif neighbor_system isa RigidBodySystem &&
               !isnothing(neighbor_system.contact_model)
            pair_parameters = rigid_contact_pair_parameters(contact_model,
                                                            neighbor_system.contact_model)
            if has_tangential_contact(pair_parameters) &&
               pair_parameters.tangential_stiffness > 0
                return true
            end
        end
    end

    return false
end

@inline function normal_friction_reference_force(contact_model,
                                                 penetration, normal_velocity)
    # `normal_velocity < 0` means approaching contact, so the dashpot contribution is
    # positive while bodies approach. Clamping prevents an attractive contact force.
    elastic_force = contact_model.normal_stiffness * penetration
    damping_force = -contact_model.normal_damping * normal_velocity

    return max(elastic_force + damping_force, zero(elastic_force))
end

@inline function contact_is_sliding(contact_model, force_trial, tangential_velocity,
                                    normal_force_friction_reference, was_sliding)
    static_limit = contact_model.static_friction_coefficient *
                   normal_force_friction_reference
    # Return mapping places the spring inside the static cone even during sliding. Retain
    # that mode while slip continues in the restoring-force direction; stopping or reversing
    # allows the spring to stick and unload before breaking away again.
    return norm(force_trial) > static_limit ||
           (was_sliding &&
            norm(tangential_velocity) > contact_model.stick_velocity_tolerance &&
            dot(force_trial, tangential_velocity) < 0)
end

function tangential_contact_force(contact_model,
                                  tangential_displacement,
                                  tangential_velocity,
                                  normal_force_friction_reference; sliding=false)
    # First evaluate the tangential spring-dashpot law. It represents sticking while its
    # magnitude remains inside the static Coulomb cone.
    force_trial = -contact_model.tangential_stiffness * tangential_displacement -
                  contact_model.tangential_damping * tangential_velocity

    trial_norm = norm(force_trial)
    if !contact_is_sliding(contact_model, force_trial, tangential_velocity,
                           normal_force_friction_reference, sliding)
        return force_trial
    end

    kinetic_limit = contact_model.kinetic_friction_coefficient *
                    normal_force_friction_reference
    kinetic_limit <= zero(kinetic_limit) && return zero(force_trial)

    tangential_speed = norm(tangential_velocity)

    if tangential_speed > zero(tangential_speed)
        # During slip, kinetic friction opposes current motion. `tanh` removes the force
        # discontinuity at zero speed without introducing an eltype-dependent velocity scale.
        regularization_velocity = contact_model.stick_velocity_tolerance
        speed_factor = regularization_velocity > zero(regularization_velocity) ?
                       tanh(tangential_speed / regularization_velocity) :
                       one(tangential_speed)
        return -kinetic_limit * speed_factor * tangential_velocity / tangential_speed
    end

    if trial_norm > zero(trial_norm)
        # At exactly zero slip speed there is no velocity direction. Preserve the restoring
        # direction of the trial force instead of reversing it.
        return kinetic_limit * force_trial / trial_norm
    end

    return zero(force_trial)
end

@inline function rigid_contact_relative_velocity(v_system, v_neighbor, system,
                                                 neighbor_system, particle, neighbor,
                                                 pos_diff)
    # Both bodies exert their contact force at the pair midpoint. Evaluate slip at that
    # same point so the contact work agrees with the translational and rotational work.
    half_separation = pos_diff / 2
    particle_velocity = current_velocity(v_system, system, particle) -
                        cross_product(system.angular_velocity[], half_separation)
    neighbor_velocity = current_velocity(v_neighbor, neighbor_system, neighbor) +
                        cross_product(neighbor_system.angular_velocity[], half_separation)
    return particle_velocity - neighbor_velocity
end

update_rigid_contact_eachstep!(system, v_ode, u_ode, semi, t, history_dt) = false

# Advance persistent contact state once after an accepted step. The Boolean return reports
# whether the force law changed, allowing the callback to invalidate an FSAL derivative only
# when necessary.
function update_rigid_contact_eachstep!(system::RigidBodySystem{<:Any, <:Any, NDIMS},
                                        v_ode, u_ode, semi, t, history_dt) where {NDIMS}
    requires_update_callback(system, semi) || return false

    v_system = wrap_v(v_ode, system, semi)
    u_system = wrap_u(u_ode, system, semi)
    active_contact_keys = Set{RigidContactKey}()
    history_changed = false

    foreach_system(semi) do neighbor_system
        neighbor_system === system && return
        has_system_interaction(system, neighbor_system, semi) || return
        history_changed |= update_contact_history_pair!(system, neighbor_system,
                                                        v_system, u_system,
                                                        v_ode, u_ode, semi, history_dt,
                                                        active_contact_keys)
    end

    contact_map = system.cache.contact_tangential_displacement
    # A key not rediscovered at the accepted endpoint no longer represents an active contact.
    # Removing it prevents stale static-friction memory from reappearing after separation.
    for key in collect(keys(contact_map))
        key in active_contact_keys && continue
        delete!(contact_map, key)
        delete!(system.cache.contact_sliding, key)
        history_changed = true
    end

    descriptor_map = system.cache.wall_contact_descriptors
    for key in collect(keys(descriptor_map))
        key in active_contact_keys && continue
        delete!(descriptor_map, key)
    end

    return history_changed
end

function update_contact_history_pair!(system, neighbor_system, v_system, u_system, v_ode,
                                      u_ode,
                                      semi, dt, active_contact_keys)
    return false
end

function update_contact_history_pair!(system::RigidBodySystem{<:Any, <:Any, NDIMS},
                                      neighbor_system::WallBoundarySystem,
                                      v_system, u_system,
                                      v_ode, u_ode,
                                      semi, dt,
                                      active_contact_keys) where {NDIMS}
    contact_model = system.contact_model
    isnothing(contact_model) && return false

    history_changed = false

    # Rebuild exactly the same transient manifolds used by the RHS, now at the accepted
    # endpoint. Only this callback pass is allowed to update persistent descriptors.
    set_zero!(system.cache.contact_manifold_count)
    set_zero!(system.cache.contact_manifold_weight_sum)
    set_zero!(system.cache.contact_manifold_penetration_sum)
    set_zero!(system.cache.contact_manifold_normal_sum)
    set_zero!(system.cache.contact_manifold_wall_velocity_sum)
    set_zero!(system.cache.contact_manifold_wall_position_sum)
    set_zero!(system.cache.contact_manifold_history_id)

    v_neighbor = wrap_v(v_ode, neighbor_system, semi)
    u_neighbor = wrap_u(u_ode, neighbor_system, semi)
    system_coords = current_coordinates(u_system, system)
    neighbor_coords = current_coordinates(u_neighbor, neighbor_system)

    foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords, semi;
                           points=each_integrated_particle(system),
                           parallelization_backend=SerialBackend()) do particle, neighbor,
                                                                       pos_diff, distance
        accumulate_wall_contact_pair!(system, v_neighbor, u_neighbor, neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      contact_model)
    end

    neighbor_system_index = system_indices(neighbor_system, semi)
    match_wall_contact_manifolds!(system, neighbor_system_index, contact_model;
                                  update_descriptors=true)
    ELTYPE = eltype(system)
    zero_tangential = zero(SVector{NDIMS, ELTYPE})

    for particle in each_integrated_particle(system)
        n_manifolds = system.cache.contact_manifold_count[particle]
        n_manifolds == 0 && continue

        particle_velocity = current_velocity(v_system, system, particle)

        for manifold_index in 1:n_manifolds
            weight_sum = system.cache.contact_manifold_weight_sum[manifold_index, particle]
            weight_sum <= eps(ELTYPE) && continue

            normal = extract_svector(system.cache.contact_manifold_normal_sum, Val(NDIMS),
                                     manifold_index, particle) / weight_sum
            normal_norm = norm(normal)
            normal_norm <= eps(ELTYPE) && continue
            normal /= normal_norm

            wall_velocity = extract_svector(system.cache.contact_manifold_wall_velocity_sum,
                                            Val(NDIMS), manifold_index, particle) /
                            weight_sum
            penetration_effective = system.cache.contact_manifold_penetration_sum[manifold_index,
                                                                                  particle] /
                                    weight_sum
            relative_velocity = particle_velocity - wall_velocity
            normal_velocity = dot(relative_velocity, normal)
            tangential_velocity = relative_velocity - normal_velocity * normal

            contact_id = system.cache.contact_manifold_history_id[manifold_index, particle]
            contact_id == 0 && continue
            contact_key = wall_contact_key(neighbor_system_index, particle, contact_id)
            push!(active_contact_keys, contact_key)
            history_changed |= update_contact_tangential_history!(system, contact_key,
                                                                  tangential_velocity,
                                                                  normal,
                                                                  penetration_effective,
                                                                  normal_velocity, dt,
                                                                  contact_model,
                                                                  zero_tangential)
        end
    end

    return history_changed
end

function match_wall_contact_manifolds!(system::RigidBodySystem{<:Any, <:Any, NDIMS},
                                       neighbor_system_index,
                                       contact_model;
                                       update_descriptors) where {NDIMS}
    descriptor_map = system.cache.wall_contact_descriptors
    isnothing(descriptor_map) && return system

    ELTYPE = eltype(system)
    normal_match_cos = convert(ELTYPE, 0.5)
    anchor_match_distance = contact_model.contact_distance
    history_ids = system.cache.contact_manifold_history_id

    # Manifold slots depend on wall-neighbor traversal order and therefore cannot identify a
    # physical contact across steps. Match each current manifold one-to-one against accepted
    # descriptors for the same rigid particle and wall system. A candidate must remain within
    # one contact distance and within 60 degrees of the accepted normal. The score balances
    # normal alignment against normalized anchor distance; the ID breaks ties deterministically.
    for particle in each_integrated_particle(system)
        n_manifolds = system.cache.contact_manifold_count[particle]
        for manifold_index in 1:n_manifolds
            weight_sum = system.cache.contact_manifold_weight_sum[manifold_index, particle]
            weight_sum <= eps(ELTYPE) && continue

            normal = extract_svector(system.cache.contact_manifold_normal_sum, Val(NDIMS),
                                     manifold_index, particle) / weight_sum
            normal_norm = norm(normal)
            normal_norm <= eps(ELTYPE) && continue
            normal /= normal_norm
            anchor = extract_svector(system.cache.contact_manifold_wall_position_sum,
                                     Val(NDIMS), manifold_index, particle) / weight_sum

            best_key = nothing
            best_score = -typemax(ELTYPE)
            for (key, descriptor) in descriptor_map
                key.neighbor_system_index == neighbor_system_index || continue
                key.local_particle == particle || continue

                already_matched = false
                for previous_manifold in 1:(manifold_index - 1)
                    if history_ids[previous_manifold, particle] == key.contact_slot
                        already_matched = true
                        break
                    end
                end
                already_matched && continue

                normal_alignment = dot(normal, descriptor.normal)
                normal_alignment >= normal_match_cos || continue
                anchor_distance = norm(anchor - descriptor.anchor)
                anchor_distance <= anchor_match_distance || continue

                score = normal_alignment - anchor_distance / anchor_match_distance
                if score > best_score ||
                   (score == best_score &&
                    (isnothing(best_key) || key.contact_slot < best_key.contact_slot))
                    best_key = key
                    best_score = score
                end
            end

            if isnothing(best_key)
                # RHS evaluations are read-only (`update_descriptors=false`): an intermediate
                # Runge-Kutta stage must never create accepted-step history. The callback
                # allocates IDs monotonically so a new contact cannot inherit stale memory.
                update_descriptors || continue
                contact_id = system.cache.next_wall_contact_id[]
                system.cache.next_wall_contact_id[] += 1
                best_key = wall_contact_key(neighbor_system_index, particle, contact_id)
            end

            history_ids[manifold_index, particle] = best_key.contact_slot
            if update_descriptors
                descriptor_map[best_key] = WallContactDescriptor(anchor, normal)
            end
        end
    end

    return system
end

function update_contact_history_pair!(system::RigidBodySystem{<:Any, <:Any, NDIMS},
                                      neighbor_system::RigidBodySystem,
                                      v_system, u_system,
                                      v_ode, u_ode,
                                      semi, dt,
                                      active_contact_keys) where {NDIMS}
    contact_model = system.contact_model
    neighbor_contact_model = neighbor_system.contact_model
    if isnothing(contact_model) || isnothing(neighbor_contact_model)
        return false
    end

    pair_parameters = rigid_contact_pair_parameters(contact_model, neighbor_contact_model)
    # Tangential damping is instantaneous. Only a nonzero pair spring needs displacement
    # history and therefore work in the accepted-step callback.
    if !has_tangential_contact(pair_parameters) ||
       pair_parameters.tangential_stiffness <= 0
        return false
    end

    history_changed = false

    v_neighbor = wrap_v(v_ode, neighbor_system, semi)
    u_neighbor = wrap_u(u_ode, neighbor_system, semi)
    system_coords = current_coordinates(u_system, system)
    neighbor_coords = current_coordinates(u_neighbor, neighbor_system)

    neighbor_system_index = system_indices(neighbor_system, semi)
    ELTYPE = eltype(system)
    zero_tangential = zero(SVector{NDIMS, ELTYPE})

    foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords, semi;
                           points=each_integrated_particle(system),
                           parallelization_backend=SerialBackend()) do particle, neighbor,
                                                                       pos_diff, distance
        distance <= eps(ELTYPE) && return

        penetration = pair_parameters.contact_distance - distance
        penetration_effective = penetration - pair_parameters.penetration_slop
        penetration_effective <= 0 && return

        normal = pos_diff / distance
        relative_velocity = rigid_contact_relative_velocity(v_system, v_neighbor, system,
                                                            neighbor_system, particle,
                                                            neighbor, pos_diff)
        normal_velocity = dot(relative_velocity, normal)
        tangential_velocity = relative_velocity - normal_velocity * normal

        contact_key = rigid_rigid_contact_key(neighbor_system_index, particle, neighbor)
        push!(active_contact_keys, contact_key)
        history_changed |= update_contact_tangential_history!(system, contact_key,
                                                              tangential_velocity,
                                                              normal,
                                                              penetration_effective,
                                                              normal_velocity, dt,
                                                              pair_parameters,
                                                              zero_tangential)
    end

    return history_changed
end

function update_contact_tangential_history!(system::RigidBodySystem, contact_key,
                                            tangential_velocity, normal,
                                            penetration_effective, normal_velocity, dt,
                                            contact_model,
                                            zero_tangential)
    contact_map = system.cache.contact_tangential_displacement
    isnothing(contact_map) && return false

    dt_ = isfinite(dt) && dt > 0 ? convert(eltype(system), dt) : zero(eltype(system))
    old_tangential_displacement = get(contact_map, contact_key, zero_tangential)
    sliding_map = system.cache.contact_sliding
    was_sliding = get(sliding_map, contact_key, false)
    sliding = false
    tangential_displacement = old_tangential_displacement

    # Integrate only accepted-step slip, then rotate old history into the current contact
    # plane. Initialization passes `dt == 0`, which registers contact identities without
    # inventing displacement before the first accepted step.
    tangential_displacement += dt_ * tangential_velocity
    tangential_displacement -= dot(tangential_displacement, normal) * normal

    if contact_model.tangential_stiffness > 0
        normal_force_reference = normal_friction_reference_force(contact_model,
                                                                 penetration_effective,
                                                                 normal_velocity)
        force_trial = -contact_model.tangential_stiffness * tangential_displacement -
                      contact_model.tangential_damping * tangential_velocity
        sliding = contact_is_sliding(contact_model, force_trial, tangential_velocity,
                                     normal_force_reference, was_sliding)

        if sliding && dt_ > 0
            # Back-calculate the spring extension from the actual kinetic force, including
            # the dashpot contribution. Keep the sliding flag so the bounded extension does
            # not incorrectly select static friction on the next RHS evaluation.
            tangential_force = tangential_contact_force(contact_model,
                                                        tangential_displacement,
                                                        tangential_velocity,
                                                        normal_force_reference;
                                                        sliding=was_sliding)
            tangential_displacement = -(tangential_force +
                                        contact_model.tangential_damping *
                                        tangential_velocity) /
                                      contact_model.tangential_stiffness
        end
    else
        tangential_displacement = zero_tangential
    end

    contact_map[contact_key] = tangential_displacement
    sliding_map[contact_key] = sliding

    return tangential_displacement != old_tangential_displacement || sliding != was_sliding
end

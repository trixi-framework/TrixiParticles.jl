@inline function requires_update_callback(contact_model::RigidContactModel)
    return contact_model.tangential_stiffness > 0 &&
           contact_model.static_friction_coefficient > 0
end

function create_cache_contact_history(contact_model::RigidContactModel, ::Val{NDIMS},
                                      ::Type{ELTYPE}) where {NDIMS, ELTYPE}
    contact_tangential_displacement = requires_update_callback(contact_model) ?
                                      Dict{RigidContactKey, SVector{NDIMS, ELTYPE}}() :
                                      nothing

    return (; contact_tangential_displacement)
end

@inline function requires_update_callback(system::RigidBodySystem)
    return !isnothing(system.contact_model) &&
           requires_update_callback(system.contact_model)
end

@inline function normal_friction_reference_force(contact_model::RigidContactModel,
                                                 penetration, normal_velocity)
    elastic_force = contact_model.normal_stiffness * penetration
    damping_force = -contact_model.normal_damping * normal_velocity

    return max(elastic_force + damping_force, zero(elastic_force))
end

function tangential_contact_force(contact_model::RigidContactModel,
                                  tangential_displacement,
                                  tangential_velocity,
                                  normal_force_friction_reference)
    force_trial = -contact_model.tangential_stiffness * tangential_displacement -
                  contact_model.tangential_damping * tangential_velocity

    trial_norm = norm(force_trial)
    static_limit = contact_model.static_friction_coefficient *
                   normal_force_friction_reference
    if trial_norm <= static_limit
        return force_trial
    end

    kinetic_limit = contact_model.kinetic_friction_coefficient *
                    normal_force_friction_reference
    kinetic_limit <= eps(eltype(tangential_velocity)) && return zero(force_trial)

    tangential_speed = norm(tangential_velocity)
    regularization_velocity = max(contact_model.stick_velocity_tolerance,
                                  sqrt(eps(eltype(tangential_velocity))))

    if tangential_speed > eps(eltype(tangential_velocity))
        speed_factor = tanh(tangential_speed / regularization_velocity)
        return -kinetic_limit * speed_factor * tangential_velocity / tangential_speed
    end

    if trial_norm > eps(eltype(tangential_velocity))
        return -kinetic_limit * force_trial / trial_norm
    end

    return zero(force_trial)
end

update_rigid_contact_eachstep!(system, v_ode, u_ode, semi, t, integrator) = system

function update_rigid_contact_eachstep!(system::RigidBodySystem{<:Any, <:Any, NDIMS},
                                        v_ode, u_ode, semi, t,
                                        integrator) where {NDIMS}
    requires_update_callback(system) || return system

    v_system = wrap_v(v_ode, system, semi)
    u_system = wrap_u(u_ode, system, semi)
    active_contact_keys = Set{RigidContactKey}()

    foreach_system(semi) do neighbor_system
        neighbor_system === system && return
        update_contact_history_pair!(system, neighbor_system, v_system, u_system,
                                     v_ode, u_ode, semi, integrator.dt,
                                     active_contact_keys)
    end

    contact_map = system.cache.contact_tangential_displacement
    for key in collect(keys(contact_map))
        key in active_contact_keys && continue
        delete!(contact_map, key)
    end

    return system
end

update_contact_history_pair!(system, neighbor_system, v_system, u_system, v_ode, u_ode,
                             semi, dt, active_contact_keys) = active_contact_keys

function update_contact_history_pair!(system::RigidBodySystem{<:Any, <:Any, NDIMS},
                                      neighbor_system::WallBoundarySystem,
                                      v_system, u_system,
                                      v_ode, u_ode,
                                      semi, dt,
                                      active_contact_keys) where {NDIMS}
    contact_model = system.contact_model
    isnothing(contact_model) && return active_contact_keys

    set_zero!(system.cache.contact_manifold_count)
    set_zero!(system.cache.contact_manifold_weight_sum)
    set_zero!(system.cache.contact_manifold_penetration_sum)
    set_zero!(system.cache.contact_manifold_normal_sum)
    set_zero!(system.cache.contact_manifold_wall_velocity_sum)

    v_neighbor = wrap_v(v_ode, neighbor_system, semi)
    u_neighbor = wrap_u(u_ode, neighbor_system, semi)
    system_coords = current_coordinates(u_system, system)
    neighbor_coords = current_coordinates(u_neighbor, neighbor_system)

    foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords, semi;
                           points=each_integrated_particle(system),
                           parallelization_backend=SerialBackend()) do particle, neighbor,
                                                                       pos_diff, distance
        accumulate_wall_contact_pair!(system, v_neighbor, neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      contact_model)
    end

    neighbor_system_index = system_indices(neighbor_system, semi)
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

            contact_key = wall_contact_key(neighbor_system_index, particle, manifold_index)
            push!(active_contact_keys, contact_key)
            update_contact_tangential_history!(system, contact_key, tangential_velocity,
                                               normal, penetration_effective,
                                               normal_velocity, dt, contact_model,
                                               zero_tangential)
        end
    end

    return active_contact_keys
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
        return active_contact_keys
    end

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

        penetration = max(contact_model.contact_distance,
                          neighbor_contact_model.contact_distance) - distance
        penetration_effective = penetration - contact_model.penetration_slop
        penetration_effective <= 0 && return

        normal = pos_diff / distance
        particle_velocity = current_velocity(v_system, system, particle)
        neighbor_velocity = current_velocity(v_neighbor, neighbor_system, neighbor)
        relative_velocity = particle_velocity - neighbor_velocity
        normal_velocity = dot(relative_velocity, normal)
        tangential_velocity = relative_velocity - normal_velocity * normal

        contact_key = rigid_rigid_contact_key(neighbor_system_index, particle, neighbor)
        push!(active_contact_keys, contact_key)
        update_contact_tangential_history!(system, contact_key, tangential_velocity,
                                           normal, penetration_effective,
                                           normal_velocity, dt, contact_model,
                                           zero_tangential)
    end

    return active_contact_keys
end

function update_contact_tangential_history!(system::RigidBodySystem, contact_key,
                                            tangential_velocity, normal,
                                            penetration_effective, normal_velocity, dt,
                                            contact_model::RigidContactModel,
                                            zero_tangential)
    contact_map = system.cache.contact_tangential_displacement
    isnothing(contact_map) && return contact_map

    dt_ = isfinite(dt) && dt > 0 ? convert(eltype(system), dt) : zero(eltype(system))
    tangential_displacement = get(contact_map, contact_key, zero_tangential)
    tangential_displacement += dt_ * tangential_velocity
    tangential_displacement -= dot(tangential_displacement, normal) * normal

    if contact_model.tangential_stiffness > eps(eltype(system))
        normal_force_reference = normal_friction_reference_force(contact_model,
                                                                 penetration_effective,
                                                                 normal_velocity)
        max_displacement = contact_model.static_friction_coefficient *
                           normal_force_reference /
                           contact_model.tangential_stiffness
        displacement_norm = norm(tangential_displacement)

        if displacement_norm > max_displacement &&
           displacement_norm > eps(eltype(system))
            tangential_displacement *= max_displacement / displacement_norm
        end
    else
        tangential_displacement = zero_tangential
    end

    contact_map[contact_key] = tangential_displacement

    return contact_map
end

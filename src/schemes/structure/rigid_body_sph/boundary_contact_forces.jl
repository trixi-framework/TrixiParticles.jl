@inline function requires_update_callback(system::RigidSPHSystem)
    return !isnothing(system.boundary_contact_model)
end

@inline function contact_time_step(system::RigidSPHSystem)
    return contact_time_step(system.boundary_contact_model, system)
end

@inline function contact_time_step(::Nothing, system::RigidSPHSystem)
    return Inf
end

function contact_time_step(contact_model::RigidBoundaryContactModel,
                           system::RigidSPHSystem)
    min_mass = minimum(system.mass)
    normal_stiffness = contact_model.normal_stiffness

    if min_mass <= eps(eltype(system)) || normal_stiffness <= eps(eltype(system))
        return Inf
    end

    return sqrt(min_mass / normal_stiffness)
end

update_rigid_contact_eachstep!(system, v_ode, u_ode, semi, t, integrator) = system

function update_contact_manifold_step!(system::RigidSPHSystem, v_ode, u_ode, semi, dt,
                                       v_system, u_system, active_contact_keys)
    foreach_system(semi) do neighbor_system
        neighbor_system isa WallBoundarySystem || return

        v_neighbor = wrap_v(v_ode, neighbor_system, semi)
        u_neighbor = wrap_u(u_ode, neighbor_system, semi)

        apply_boundary_contact_correction!(system, neighbor_system,
                                           v_system, u_system,
                                           v_neighbor, u_neighbor,
                                           semi, dt,
                                           active_contact_keys)
    end

    return active_contact_keys
end

function update_contact_history_step!(system::RigidSPHSystem, active_contact_keys)
    remove_inactive_contact_pairs!(system.cache.contact_tangential_displacement,
                                   active_contact_keys)

    return active_contact_keys
end

function apply_optional_resting_projection_step!(system::RigidSPHSystem,
                                                 v_system, u_system,
                                                 v_ode, u_ode, semi, integrator)
    state_modified = project_resting_contact_velocity!(system, v_system, u_system,
                                                       v_ode, u_ode, semi, integrator)
    state_modified && u_modified!(integrator, true)

    return state_modified
end

function update_rigid_contact_eachstep!(system::RigidSPHSystem, v_ode, u_ode, semi, t,
                                        integrator)
    contact_model = system.boundary_contact_model
    isnothing(contact_model) && return system

    v_system = wrap_v(v_ode, system, semi)
    u_system = wrap_u(u_ode, system, semi)

    system.cache.boundary_contact_count[] = 0
    system.cache.max_boundary_penetration[] = zero(eltype(system))

    active_contact_keys = Set{NTuple{3, Int}}()
    @trixi_timeit timer() "contact manifold update" begin
        update_contact_manifold_step!(system, v_ode, u_ode, semi, integrator.dt,
                                      v_system, u_system, active_contact_keys)
    end

    @trixi_timeit timer() "history update" begin
        update_contact_history_step!(system, active_contact_keys)
    end

    # Fallback for persistent resting contacts: project rigid-body velocity to enforce
    # non-penetration when the adaptive solver collapses to very small time steps.
    @trixi_timeit timer() "collision projection" begin
        apply_optional_resting_projection_step!(system,
                                                v_system, u_system,
                                                v_ode, u_ode, semi, integrator)
    end

    return system
end

function update_contact_manifold_cache!(system::RigidSPHSystem{<:Any, <:Any, NDIMS},
                                        neighbor_system::WallBoundarySystem,
                                        u_system,
                                        v_neighbor, u_neighbor,
                                        semi,
                                        contact_model::RigidBoundaryContactModel,
                                        normal_merge_cos, zero_tangential,
                                        ELTYPE) where {NDIMS}
    system_coords = current_coordinates(u_system, system)
    neighbor_coords = current_coordinates(u_neighbor, neighbor_system)

    foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords, semi;
                           points=each_integrated_particle(system),
                           parallelization_backend=SerialBackend()) do particle, neighbor,
                                                                       pos_diff, distance
        distance <= eps(ELTYPE) && return

        penetration = contact_model.contact_distance - distance
        penetration_effective = penetration - contact_model.penetration_slop
        penetration_effective <= 0 && return

        normal = pos_diff / distance
        wall_velocity = current_velocity(v_neighbor, neighbor_system, neighbor)
        contact_weight = wall_contact_pair_weight(neighbor_system, distance, neighbor, ELTYPE)
        contact_weight <= eps(ELTYPE) && return

        manifold = find_or_add_contact_manifold!(system.cache, particle, normal,
                                                 normal_merge_cos, ELTYPE)
        accumulate_contact_manifold!(system.cache, particle, manifold, contact_weight,
                                     normal, wall_velocity, penetration_effective,
                                     zero_tangential)
    end

    return system.cache
end

function update_contact_history_from_manifolds!(system::RigidSPHSystem{<:Any, <:Any, NDIMS},
                                                neighbor_system_index,
                                                v_system, dt,
                                                active_contact_keys,
                                                contact_model::RigidBoundaryContactModel,
                                                ELTYPE) where {NDIMS}
    contact_count = 0
    max_penetration = zero(ELTYPE)

    for particle in each_integrated_particle(system)
        n_manifolds = system.cache.contact_manifold_count[particle]
        n_manifolds == 0 && continue

        particle_velocity = current_velocity(v_system, system, particle)

        for manifold in 1:n_manifolds
            weight_sum = system.cache.contact_manifold_weight_sum[manifold, particle]
            weight_sum <= eps(ELTYPE) && continue

            normal = weighted_manifold_vector(system.cache.contact_manifold_normal_sum,
                                              manifold, particle, weight_sum, Val(NDIMS),
                                              ELTYPE)
            normal_norm = norm(normal)
            normal_norm <= eps(ELTYPE) && continue
            normal /= normal_norm

            wall_velocity = weighted_manifold_vector(system.cache.contact_manifold_wall_velocity_sum,
                                                     manifold, particle, weight_sum,
                                                     Val(NDIMS), ELTYPE)
            penetration_effective = system.cache.contact_manifold_penetration_sum[manifold,
                                                                                   particle] /
                                    weight_sum
            relative_velocity = particle_velocity - wall_velocity
            normal_velocity = dot(relative_velocity, normal)
            tangential_velocity = relative_velocity - normal_velocity * normal

            contact_key = (neighbor_system_index, particle, manifold)
            push!(active_contact_keys, contact_key)
            update_contact_tangential_history!(system, contact_key, tangential_velocity,
                                               normal, penetration_effective,
                                               normal_velocity, dt,
                                               contact_model)

            contact_count += 1
            max_penetration = max(max_penetration, penetration_effective)
        end
    end

    return contact_count, max_penetration
end

function apply_boundary_contact_correction!(system::RigidSPHSystem{<:Any, <:Any, NDIMS},
                                            neighbor_system::WallBoundarySystem,
                                            v_system, u_system,
                                            v_neighbor, u_neighbor,
                                            semi, dt, active_contact_keys) where {NDIMS}
    contact_model = system.boundary_contact_model
    isnothing(contact_model) && return false

    reset_contact_manifold_cache!(system.cache)
    neighbor_system_index = system_indices(neighbor_system, semi)
    ELTYPE = eltype(system)
    zero_tangential = zero(SVector{NDIMS, ELTYPE})
    normal_merge_cos = convert(ELTYPE, 0.995)

    update_contact_manifold_cache!(system, neighbor_system,
                                   u_system,
                                   v_neighbor, u_neighbor,
                                   semi, contact_model,
                                   normal_merge_cos, zero_tangential,
                                   ELTYPE)
    contact_count, max_penetration = update_contact_history_from_manifolds!(system,
                                                                            neighbor_system_index,
                                                                            v_system, dt,
                                                                            active_contact_keys,
                                                                            contact_model,
                                                                            ELTYPE)

    system.cache.boundary_contact_count[] += contact_count
    system.cache.max_boundary_penetration[] = max(system.cache.max_boundary_penetration[],
                                                  max_penetration)

    # This callback only updates contact history/diagnostics. It must not modify
    # per-particle states directly because that violates rigid-body kinematics.
    return false
end

function update_contact_tangential_history!(system::RigidSPHSystem, contact_key,
                                            tangential_velocity, normal,
                                            penetration_effective, normal_velocity, dt,
                                            contact_model::RigidBoundaryContactModel)
    dt_ = isfinite(dt) && dt > 0 ? convert(eltype(system), dt) : zero(eltype(system))
    contact_map = system.cache.contact_tangential_displacement
    isnothing(contact_map) && return contact_map
    tangential_displacement = get(contact_map, contact_key,
                                  zero(SVector{ndims(system), eltype(system)}))

    tangential_displacement += dt_ * tangential_velocity
    tangential_displacement -= dot(tangential_displacement, normal) * normal

    if contact_model.tangential_stiffness > eps(eltype(system))
        normal_force = normal_friction_reference_force(contact_model, penetration_effective,
                                                       normal_velocity, eltype(system))
        max_displacement = contact_model.static_friction_coefficient * normal_force /
                           contact_model.tangential_stiffness
        displacement_norm = norm(tangential_displacement)

        if displacement_norm > max_displacement && displacement_norm > eps(eltype(system))
            tangential_displacement *= max_displacement / displacement_norm
        end
    else
        tangential_displacement = zero(tangential_displacement)
    end

    contact_map[contact_key] = tangential_displacement

    return contact_map
end

function remove_inactive_contact_pairs!(contact_tangential_displacement, active_contact_keys)
    isnothing(contact_tangential_displacement) && return contact_tangential_displacement

    for key in collect(keys(contact_tangential_displacement))
        key in active_contact_keys && continue
        delete!(contact_tangential_displacement, key)
    end

    return contact_tangential_displacement
end

struct RigidBoundaryContactModel{ELTYPE <: Real}
    normal_stiffness::ELTYPE
    normal_damping::ELTYPE
    static_friction_coefficient::ELTYPE
    kinetic_friction_coefficient::ELTYPE
    tangential_stiffness::ELTYPE
    tangential_damping::ELTYPE
    contact_distance::ELTYPE
    stick_velocity_tolerance::ELTYPE
    penetration_slop::ELTYPE
    torque_free::Bool
    resting_contact_projection::Bool
end

function RigidBoundaryContactModel(; normal_stiffness,
                                   normal_damping=0.0,
                                   static_friction_coefficient=0.5,
                                   kinetic_friction_coefficient=0.4,
                                   tangential_stiffness=0.0,
                                   tangential_damping=0.0,
                                   contact_distance=0.0,
                                   stick_velocity_tolerance=1e-6,
                                   penetration_slop=0.0,
                                   torque_free=false,
                                   resting_contact_projection=true)
    ELTYPE = promote_type(typeof(normal_stiffness),
                          typeof(normal_damping),
                          typeof(static_friction_coefficient),
                          typeof(kinetic_friction_coefficient),
                          typeof(tangential_stiffness),
                          typeof(tangential_damping),
                          typeof(contact_distance),
                          typeof(stick_velocity_tolerance),
                          typeof(penetration_slop))

    normal_stiffness_ = convert(ELTYPE, normal_stiffness)
    normal_damping_ = convert(ELTYPE, normal_damping)
    static_friction_coefficient_ = convert(ELTYPE, static_friction_coefficient)
    kinetic_friction_coefficient_ = convert(ELTYPE, kinetic_friction_coefficient)
    tangential_stiffness_ = convert(ELTYPE, tangential_stiffness)
    tangential_damping_ = convert(ELTYPE, tangential_damping)
    contact_distance_ = convert(ELTYPE, contact_distance)
    stick_velocity_tolerance_ = convert(ELTYPE, stick_velocity_tolerance)
    penetration_slop_ = convert(ELTYPE, penetration_slop)

    normal_stiffness_ > 0 ||
        throw(ArgumentError("`normal_stiffness` must be positive"))
    normal_damping_ >= 0 ||
        throw(ArgumentError("`normal_damping` must be non-negative"))
    static_friction_coefficient_ >= 0 ||
        throw(ArgumentError("`static_friction_coefficient` must be non-negative"))
    kinetic_friction_coefficient_ >= 0 ||
        throw(ArgumentError("`kinetic_friction_coefficient` must be non-negative"))
    kinetic_friction_coefficient_ <= static_friction_coefficient_ ||
        throw(ArgumentError("`kinetic_friction_coefficient` must be <= `static_friction_coefficient`"))
    tangential_stiffness_ >= 0 ||
        throw(ArgumentError("`tangential_stiffness` must be non-negative"))
    tangential_damping_ >= 0 ||
        throw(ArgumentError("`tangential_damping` must be non-negative"))
    contact_distance_ >= 0 ||
        throw(ArgumentError("`contact_distance` must be non-negative"))
    stick_velocity_tolerance_ >= 0 ||
        throw(ArgumentError("`stick_velocity_tolerance` must be non-negative"))
    penetration_slop_ >= 0 ||
        throw(ArgumentError("`penetration_slop` must be non-negative"))

    return RigidBoundaryContactModel(normal_stiffness_, normal_damping_,
                                     static_friction_coefficient_,
                                     kinetic_friction_coefficient_,
                                     tangential_stiffness_, tangential_damping_,
                                     contact_distance_, stick_velocity_tolerance_,
                                     penetration_slop_, Bool(torque_free),
                                     Bool(resting_contact_projection))
end

function convert_boundary_contact_model(::Nothing, particle_spacing, ELTYPE)
    return nothing
end

function convert_boundary_contact_model(model::RigidBoundaryContactModel, particle_spacing,
                                        ELTYPE)
    contact_distance = model.contact_distance > 0 ? model.contact_distance :
                       convert(ELTYPE, particle_spacing)

    return RigidBoundaryContactModel(; normal_stiffness=convert(ELTYPE,
                                                                model.normal_stiffness),
                                     normal_damping=convert(ELTYPE, model.normal_damping),
                                     static_friction_coefficient=convert(ELTYPE,
                                                                         model.static_friction_coefficient),
                                     kinetic_friction_coefficient=convert(ELTYPE,
                                                                          model.kinetic_friction_coefficient),
                                     tangential_stiffness=convert(ELTYPE,
                                                                  model.tangential_stiffness),
                                     tangential_damping=convert(ELTYPE,
                                                                model.tangential_damping),
                                     contact_distance,
                                     stick_velocity_tolerance=convert(ELTYPE,
                                                                      model.stick_velocity_tolerance),
                                     penetration_slop=convert(ELTYPE,
                                                              model.penetration_slop),
                                     torque_free=model.torque_free,
                                     resting_contact_projection=model.resting_contact_projection)
end

create_contact_tangential_displacement(::Nothing, ELTYPE, ::Val{NDIMS}) where {NDIMS} = nothing

function create_contact_tangential_displacement(::RigidBoundaryContactModel, ELTYPE,
                                                ::Val{NDIMS}) where {NDIMS}
    return Dict{NTuple{3, Int}, SVector{NDIMS, ELTYPE}}()
end

function Base.show(io::IO, model::RigidBoundaryContactModel)
    @nospecialize model # reduce precompilation time

    print(io, "RigidBoundaryContactModel(")
    print(io, "normal_stiffness=", model.normal_stiffness)
    print(io, ", normal_damping=", model.normal_damping)
    print(io, ", mu_s=", model.static_friction_coefficient)
    print(io, ", mu_k=", model.kinetic_friction_coefficient)
    print(io, ", tangential_stiffness=", model.tangential_stiffness)
    print(io, ", tangential_damping=", model.tangential_damping)
    print(io, ", contact_distance=", model.contact_distance)
    print(io, ", stick_velocity_tolerance=", model.stick_velocity_tolerance)
    print(io, ", penetration_slop=", model.penetration_slop)
    print(io, ", torque_free=", model.torque_free)
    print(io, ", resting_contact_projection=", model.resting_contact_projection)
    print(io, ")")
end

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

function update_rigid_contact_eachstep!(system::RigidSPHSystem, v_ode, u_ode, semi, t,
                                        integrator)
    contact_model = system.boundary_contact_model
    isnothing(contact_model) && return system

    v_system = wrap_v(v_ode, system, semi)
    u_system = wrap_u(u_ode, system, semi)

    system.cache.boundary_contact_count[] = 0
    system.cache.max_boundary_penetration[] = zero(eltype(system))

    active_contact_keys = Set{NTuple{3, Int}}()
    @trixi_timeit timer() "update collision history" begin
        foreach_system(semi) do neighbor_system
            if neighbor_system isa WallBoundarySystem
                v_neighbor = wrap_v(v_ode, neighbor_system, semi)
                u_neighbor = wrap_u(u_ode, neighbor_system, semi)

                apply_boundary_contact_correction!(system, neighbor_system,
                                                   v_system, u_system,
                                                   v_neighbor, u_neighbor,
                                                   semi, integrator.dt,
                                                   active_contact_keys)
            end
        end
    end

    remove_inactive_contact_pairs!(system.cache.contact_tangential_displacement,
                                   active_contact_keys)

    # Fallback for persistent resting contacts: project rigid-body velocity to enforce
    # non-penetration when the adaptive solver collapses to very small time steps.
    state_modified = @trixi_timeit timer() "collision projection" begin
        project_resting_contact_velocity!(system, v_system, u_system,
                                          v_ode, u_ode, semi, integrator)
    end
    state_modified && u_modified!(integrator, true)

    return system
end

function apply_boundary_contact_correction!(system::RigidSPHSystem{<:Any, <:Any, NDIMS},
                                            neighbor_system::WallBoundarySystem,
                                            v_system, u_system,
                                            v_neighbor, u_neighbor,
                                            semi, dt, active_contact_keys) where {NDIMS}
    contact_model = system.boundary_contact_model
    isnothing(contact_model) && return false

    reset_contact_manifold_cache!(system.cache)

    system_coords = current_coordinates(u_system, system)
    neighbor_coords = current_coordinates(u_neighbor, neighbor_system)
    neighbor_system_index = system_indices(neighbor_system, semi)
    ELTYPE = eltype(system)
    zero_tangential = zero(SVector{NDIMS, ELTYPE})
    normal_merge_cos = convert(ELTYPE, 0.995)

    contact_count = 0
    max_penetration = zero(ELTYPE)
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

project_resting_contact_velocity!(system, v_system, u_system, v_ode, u_ode, semi,
                                  integrator) = false

@inline function rotational_velocity_2d(angular_velocity, relative_position)
    return SVector(-angular_velocity * relative_position[2],
                   angular_velocity * relative_position[1])
end

function add_or_merge_resting_constraint!(constraints,
                                          particle, normal, wall_velocity,
                                          penetration_effective, normal_merge_cos)
    for i in eachindex(constraints)
        existing_particle, existing_normal, _, existing_penetration = constraints[i]
        if existing_particle == particle && dot(existing_normal, normal) >= normal_merge_cos
            if penetration_effective > existing_penetration
                constraints[i] = (particle, normal, wall_velocity, penetration_effective)
            end
            return constraints
        end
    end

    push!(constraints, (particle, normal, wall_velocity, penetration_effective))

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

function project_resting_contact_position!(system::RigidSPHSystem{<:Any, <:Any, 2},
                                           u_system, constraints,
                                           penetration_tolerance)
    total_mass = system.cache.total_mass
    total_mass <= eps(eltype(system)) && return false

    inv_total_mass = inv(total_mass)

    center_of_mass_correction = zero(SVector{2, eltype(system)})
    any_correction = false

    for _ in 1:3
        iteration_has_correction = false

        for (particle, normal, _, penetration_effective) in constraints
            penetration_effective <= penetration_tolerance && continue

            applied_correction = center_of_mass_correction
            remaining_penetration = penetration_effective -
                                    dot(applied_correction, normal)
            remaining_penetration <= penetration_tolerance && continue

            effective_inverse_mass = inv_total_mass
            effective_inverse_mass <= eps(eltype(system)) && continue

            correction_impulse = (remaining_penetration - penetration_tolerance) /
                                 effective_inverse_mass
            correction_impulse <= eps(eltype(system)) && continue

            center_of_mass_correction += correction_impulse * inv_total_mass * normal
            iteration_has_correction = true
        end

        any_correction |= iteration_has_correction
        iteration_has_correction || break
    end

    any_correction || return false

    system_coords = current_coordinates(u_system, system)
    for particle in each_integrated_particle(system)
        @inbounds begin
            system_coords[1, particle] += center_of_mass_correction[1]
            system_coords[2, particle] += center_of_mass_correction[2]
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
    if system.cache.boundary_contact_count[] == 0
        reset_resting_contact_counter!(system)
        return false
    end

    dt = integrator.dt
    dt_contact = contact_time_step(contact_model, system)
    if !isfinite(dt) || dt <= 0 || !isfinite(dt_contact) || dt_contact <= 0
        reset_resting_contact_counter!(system)
        return false
    end

    center_of_mass_velocity = system.cache.center_of_mass_velocity[]
    angular_velocity = system.cache.angular_velocity[]
    ELTYPE = eltype(system)
    zero_tangential = zero(SVector{2, ELTYPE})
    velocity_floor = max(convert(ELTYPE, 0.1) *
                         contact_model.stick_velocity_tolerance,
                         sqrt(eps(ELTYPE)))

    system_coords = current_coordinates(u_system, system)
    constraint_type = Tuple{Int, SVector{2, eltype(system)}, SVector{2, eltype(system)},
                            eltype(system)}
    constraints = constraint_type[]
    projection_contact_keys = Set{NTuple{3, Int}}()
    projected_particles = Set{Int}()
    normal_merge_cos = convert(eltype(system), 0.995)
    max_normal_speed = zero(eltype(system))
    max_tangential_speed = zero(eltype(system))

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
                                                 penetration_effective, normal_merge_cos)
            end
        end
    end

    if isempty(constraints)
        reset_resting_contact_counter!(system)
        return false
    end

    # Use a persistence-based resting-contact trigger; keep dt-ratio as a secondary
    # early-warning signal to avoid waiting for severe adaptive-step collapse.
    resting_speed_max = resting_contact_speed_threshold(contact_model, dt_contact,
                                                        velocity_floor, ELTYPE)
    in_resting_contact = max_normal_speed <= resting_speed_max &&
                         max_tangential_speed <= resting_speed_max
    resting_counter = update_resting_contact_counter!(system, in_resting_contact)
    in_resting_contact || return false

    dt_projection_trigger = convert(ELTYPE, 0.1) * dt_contact
    projection_persistence_steps = 2
    if dt > dt_projection_trigger && resting_counter < projection_persistence_steps
        return false
    end

    total_mass = system.cache.total_mass
    total_mass <= eps(eltype(system)) && return false

    inv_total_mass = inv(total_mass)

    projected_center_of_mass_velocity = center_of_mass_velocity
    velocity_modified = false

    for _ in 1:3
        any_projection = false

        for (_, normal, wall_velocity, _) in constraints
            particle_velocity = projected_center_of_mass_velocity
            relative_velocity = particle_velocity - wall_velocity
            normal_velocity = dot(relative_velocity, normal)
            normal_velocity >= -velocity_floor && continue

            effective_inverse_mass = inv_total_mass
            effective_inverse_mass <= eps(eltype(system)) && continue

            impulse = -normal_velocity / effective_inverse_mass
            impulse <= eps(eltype(system)) && continue

            projected_center_of_mass_velocity += impulse * inv_total_mass * normal
            any_projection = true
            velocity_modified = true
        end

        any_projection || break
    end

    if norm(projected_center_of_mass_velocity) <= velocity_floor
        projected_center_of_mass_velocity = zero(projected_center_of_mass_velocity)
    end

    velocity_modified = velocity_modified ||
                        norm(projected_center_of_mass_velocity -
                             center_of_mass_velocity) > eps(ELTYPE)

    if velocity_modified
        system_velocity = current_velocity(v_system, system)
        for particle in each_integrated_particle(system)
            relative_position = extract_svector(system.cache.relative_coordinates, system, particle)
            particle_velocity = projected_center_of_mass_velocity +
                                rotational_velocity_2d(angular_velocity,
                                                       relative_position)

            @inbounds begin
                system_velocity[1, particle] = particle_velocity[1]
                system_velocity[2, particle] = particle_velocity[2]
            end
        end
    end

    contact_scale = max(contact_model.contact_distance, system.particle_spacing)
    penetration_tolerance = max(convert(ELTYPE, 1.0e-3) * contact_scale,
                                sqrt(eps(ELTYPE)) * contact_scale)
    position_modified = project_resting_contact_position!(system, u_system, constraints,
                                                          penetration_tolerance)
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

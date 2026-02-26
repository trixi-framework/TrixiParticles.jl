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
end

function RigidBoundaryContactModel(; normal_stiffness,
                                   normal_damping=0.0,
                                   static_friction_coefficient=0.5,
                                   kinetic_friction_coefficient=0.4,
                                   tangential_stiffness=0.0,
                                   tangential_damping=0.0,
                                   contact_distance=0.0,
                                   stick_velocity_tolerance=1e-6,
                                   penetration_slop=0.0)
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
                                     penetration_slop_)
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
                                                              model.penetration_slop))
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

    remove_inactive_contact_pairs!(system.cache.contact_tangential_displacement,
                                   active_contact_keys)

    return system
end

function apply_boundary_contact_correction!(system::RigidSPHSystem,
                                            neighbor_system::WallBoundarySystem,
                                            v_system, u_system,
                                            v_neighbor, u_neighbor,
                                            semi, dt, active_contact_keys)
    contact_model = system.boundary_contact_model
    isnothing(contact_model) && return false

    system_coords = current_coordinates(u_system, system)
    neighbor_coords = current_coordinates(u_neighbor, neighbor_system)
    neighbor_system_index = system_indices(neighbor_system, semi)

    contact_count = 0
    max_penetration = zero(eltype(system))
    foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords, semi;
                           points=each_integrated_particle(system),
                           parallelization_backend=SerialBackend()) do particle, neighbor,
                                                                       pos_diff, distance
        distance <= eps(eltype(system)) && return

        penetration = contact_model.contact_distance - distance
        penetration <= contact_model.penetration_slop && return

        contact_key = (neighbor_system_index, particle, neighbor)
        push!(active_contact_keys, contact_key)

        normal = pos_diff / distance

        relative_velocity = current_velocity(v_system, system, particle) -
                            current_velocity(v_neighbor, neighbor_system, neighbor)
        tangential_velocity = relative_velocity -
                              dot(relative_velocity, normal) * normal
        update_contact_tangential_history!(system, contact_key, tangential_velocity,
                                           normal, penetration, dt, contact_model)

        contact_count += 1
        max_penetration = max(max_penetration, penetration)
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
                                            penetration, dt,
                                            contact_model::RigidBoundaryContactModel)
    dt_ = isfinite(dt) && dt > 0 ? convert(eltype(system), dt) : zero(eltype(system))
    contact_map = system.cache.contact_tangential_displacement
    isnothing(contact_map) && return contact_map
    tangential_displacement = get(contact_map, contact_key,
                                  zero(SVector{ndims(system), eltype(system)}))

    tangential_displacement += dt_ * tangential_velocity
    tangential_displacement -= dot(tangential_displacement, normal) * normal

    if contact_model.tangential_stiffness > eps(eltype(system))
        normal_force = contact_model.normal_stiffness * penetration
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

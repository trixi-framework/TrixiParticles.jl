abstract type AbstractRigidContactModel end

"""
    RigidContactModel(; normal_stiffness,
                      normal_damping=0.0,
                      static_friction_coefficient=nothing,
                      kinetic_friction_coefficient=nothing,
                      tangential_stiffness=nothing,
                      tangential_damping=nothing,
                      contact_distance=0.0,
                      stick_velocity_tolerance=nothing,
                      penetration_slop=nothing,
                      torque_free=false,
                      resting_contact_projection=nothing,
                      normalize_force_by_contact_patch=false)

Runtime rigid-contact model shared by rigid-wall and rigid-rigid contact.
The currently active shared contact force on `main` is a linear normal spring-dashpot law.
The richer tangential/friction parameters and wall-specific controls are stored on the
runtime model already so the later rigid-contact porting PRs can build on the same
runtime representation without another API change.

If `contact_distance == 0`, the particle spacing of the `RigidBodySystem` will be used
as contact distance when the model is adapted via
`copy_contact_model(model, particle_spacing, ELTYPE)`.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
struct RigidContactModel{ELTYPE <: Real} <: AbstractRigidContactModel
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
    normalize_force_by_contact_patch::Bool
end

function RigidContactModel(; normal_stiffness,
                           normal_damping=0.0,
                           static_friction_coefficient=nothing,
                           kinetic_friction_coefficient=nothing,
                           tangential_stiffness=nothing,
                           tangential_damping=nothing,
                           contact_distance=0.0,
                           stick_velocity_tolerance=nothing,
                           penetration_slop=nothing,
                           torque_free=false,
                           resting_contact_projection=nothing,
                           normalize_force_by_contact_patch=false)
    advanced_mode = !isnothing(static_friction_coefficient) ||
                    !isnothing(kinetic_friction_coefficient) ||
                    !isnothing(tangential_stiffness) ||
                    !isnothing(tangential_damping) ||
                    !isnothing(stick_velocity_tolerance) ||
                    !isnothing(penetration_slop) ||
                    !isnothing(resting_contact_projection) ||
                    torque_free

    static_friction_coefficient = something(static_friction_coefficient,
                                            advanced_mode ? 0.5 : 0.0)
    kinetic_friction_coefficient = something(kinetic_friction_coefficient,
                                             advanced_mode ? 0.4 : 0.0)
    tangential_stiffness = something(tangential_stiffness, 0.0)
    tangential_damping = something(tangential_damping, 0.0)
    stick_velocity_tolerance = something(stick_velocity_tolerance, 1e-6)
    penetration_slop = something(penetration_slop, 0.0)
    resting_contact_projection = something(resting_contact_projection, advanced_mode)
    normalize_force_by_contact_patch = Bool(normalize_force_by_contact_patch)
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

    return RigidContactModel(normal_stiffness_, normal_damping_,
                             static_friction_coefficient_,
                             kinetic_friction_coefficient_,
                             tangential_stiffness_, tangential_damping_,
                             contact_distance_, stick_velocity_tolerance_,
                             penetration_slop_, Bool(torque_free),
                             Bool(resting_contact_projection),
                             normalize_force_by_contact_patch)
end

function copy_contact_model(model::RigidContactModel, particle_spacing,
                            ::Type{ELTYPE}) where {ELTYPE}
    particle_spacing_ = convert(ELTYPE, particle_spacing)
    particle_spacing_ > 0 ||
        throw(ArgumentError("`particle_spacing` must be positive"))

    contact_distance = model.contact_distance > 0 ?
                       convert(ELTYPE, model.contact_distance) :
                       particle_spacing_

    return RigidContactModel(; normal_stiffness=convert(ELTYPE, model.normal_stiffness),
                             normal_damping=convert(ELTYPE, model.normal_damping),
                             static_friction_coefficient=convert(ELTYPE,
                                                                 model.static_friction_coefficient),
                             kinetic_friction_coefficient=convert(ELTYPE,
                                                                  model.kinetic_friction_coefficient),
                             tangential_stiffness=convert(ELTYPE,
                                                          model.tangential_stiffness),
                             tangential_damping=convert(ELTYPE, model.tangential_damping),
                             contact_distance,
                             stick_velocity_tolerance=convert(ELTYPE,
                                                              model.stick_velocity_tolerance),
                             penetration_slop=convert(ELTYPE, model.penetration_slop),
                             torque_free=model.torque_free,
                             resting_contact_projection=model.resting_contact_projection,
                             normalize_force_by_contact_patch=model.normalize_force_by_contact_patch)
end

function contact_time_step(system::RigidBodySystem, semi)
    # for rigid-wall interaction, limit timestep to the single body contact time step,
    # for rigid-rigid interactions we need to check all neighbors
    dt = contact_time_step(system, system) * sqrt(2)

    # TODO this is called for every system, so we compute this twice for every interaction pair
    foreach_system(semi) do neighbor
        neighbor === system && return
        dt = min(dt, contact_time_step(system, neighbor))
    end

    return dt
end

@inline function contact_time_step(contact_model::Nothing, system::RigidBodySystem)
    return Inf
end

@inline function contact_time_step(system::RigidBodySystem,
                                   neighbor::RigidBodySystem)
    return contact_time_step(system.contact_model, system, neighbor.contact_model, neighbor)
end

@inline function contact_time_step(system, neighbor)
    return Inf
end

function contact_time_step(contact_model::RigidContactModel,
                           system::RigidBodySystem,
                           neighbor_contact_model::RigidContactModel,
                           neighbor_system::RigidBodySystem)
    pair_normal_stiffness = (contact_model.normal_stiffness +
                             neighbor_contact_model.normal_stiffness) / 2

    min_mass = minimum(system.mass)
    neighbor_min_mass = minimum(neighbor_system.mass)
    return sqrt((min_mass * neighbor_min_mass / (min_mass + neighbor_min_mass)) /
                pair_normal_stiffness)
end

function contact_time_step(contact_model,
                           system::RigidBodySystem,
                           neighbor_contact_model,
                           neighbor_system::RigidBodySystem)
    return Inf
end

function Base.show(io::IO, model::RigidContactModel)
    @nospecialize model # reduce precompilation time

    print(io, "RigidContactModel(")
    print(io, "normal_stiffness=", model.normal_stiffness)
    print(io, ", normal_damping=", model.normal_damping)
    print(io, ", static_friction_coefficient=", model.static_friction_coefficient)
    print(io, ", kinetic_friction_coefficient=", model.kinetic_friction_coefficient)
    print(io, ", tangential_stiffness=", model.tangential_stiffness)
    print(io, ", tangential_damping=", model.tangential_damping)
    print(io, ", contact_distance=", model.contact_distance)
    print(io, ", stick_velocity_tolerance=", model.stick_velocity_tolerance)
    print(io, ", penetration_slop=", model.penetration_slop)
    print(io, ", torque_free=", model.torque_free)
    print(io, ", resting_contact_projection=", model.resting_contact_projection)
    print(io, ", normalize_force_by_contact_patch=", model.normalize_force_by_contact_patch)
    print(io, ")")
end

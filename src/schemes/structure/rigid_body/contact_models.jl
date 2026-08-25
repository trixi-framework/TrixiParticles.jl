abstract type AbstractRigidContactModel end

@enum RigidContactKind::UInt8 begin
    WallContact = 1
    RigidRigidContact = 2
end

"""
    RigidContactKey(neighbor_system_index, local_particle, contact_slot, contact_kind)

Shared tangential-history key for rigid contact.

`contact_slot` stores the wall-manifold index for rigid-wall contact and the neighbor
particle index for rigid-rigid contact.
"""
struct RigidContactKey
    neighbor_system_index::Int
    local_particle::Int
    contact_slot::Int
    contact_kind::RigidContactKind
end

@inline wall_contact_key(neighbor_system_index, local_particle,
                         manifold_index) = RigidContactKey(neighbor_system_index,
                                                           local_particle, manifold_index,
                                                           WallContact)

@inline rigid_rigid_contact_key(neighbor_system_index, local_particle,
                                neighbor_particle) = RigidContactKey(neighbor_system_index,
                                                                     local_particle,
                                                                     neighbor_particle,
                                                                     RigidRigidContact)

@inline function Base.:(==)(lhs::RigidContactKey, rhs::RigidContactKey)
    return lhs.neighbor_system_index == rhs.neighbor_system_index &&
           lhs.local_particle == rhs.local_particle &&
           lhs.contact_slot == rhs.contact_slot &&
           lhs.contact_kind == rhs.contact_kind
end

@inline Base.isequal(lhs::RigidContactKey, rhs::RigidContactKey) = lhs == rhs

@inline function Base.hash(key::RigidContactKey, h::UInt)
    return hash((key.neighbor_system_index, key.local_particle,
                 key.contact_slot, key.contact_kind), h)
end

"""
    RigidContactModel(; normal_stiffness,
                      normal_damping=0.0,
                      static_friction_coefficient=nothing,
                      kinetic_friction_coefficient=nothing,
                      tangential_stiffness=nothing,
                      tangential_damping=nothing,
                      contact_distance=0.0,
                      stick_velocity_tolerance=nothing,
                      penetration_slop=nothing)

Shared rigid-contact model used by the active rigid-wall and rigid-rigid contact paths.
Rigid-wall contact currently combines the linear normal spring-dashpot law with
history-driven tangential friction updated through `UpdateCallback`, while rigid-rigid
contact remains normal-only but shares the same runtime model and history-key design.

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
end

function RigidContactModel(; normal_stiffness,
                           normal_damping=0.0,
                           static_friction_coefficient=nothing,
                           kinetic_friction_coefficient=nothing,
                           tangential_stiffness=nothing,
                           tangential_damping=nothing,
                           contact_distance=0.0,
                           stick_velocity_tolerance=nothing,
                           penetration_slop=nothing)
    tangential_mode = !isnothing(static_friction_coefficient) ||
                      !isnothing(kinetic_friction_coefficient) ||
                      !isnothing(tangential_stiffness) ||
                      !isnothing(tangential_damping)

    static_friction_coefficient = something(static_friction_coefficient,
                                            tangential_mode ? 0.5 : 0.0)
    kinetic_friction_coefficient = something(kinetic_friction_coefficient,
                                             tangential_mode ? 0.4 : 0.0)
    tangential_stiffness = something(tangential_stiffness, 0.0)
    tangential_damping = something(tangential_damping, 0.0)
    stick_velocity_tolerance = something(stick_velocity_tolerance, 1.0e-6)
    penetration_slop = something(penetration_slop, 0.0)
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
                             tangential_stiffness_,
                             tangential_damping_,
                             contact_distance_,
                             stick_velocity_tolerance_,
                             penetration_slop_)
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
                             tangential_damping=convert(ELTYPE,
                                                        model.tangential_damping),
                             contact_distance,
                             stick_velocity_tolerance=convert(ELTYPE,
                                                              model.stick_velocity_tolerance),
                             penetration_slop=convert(ELTYPE, model.penetration_slop))
end

# Single-body rigid-contact scale.
#
# This models the rigid body contacting an infinite-mass wall with its own contact model.
# It is intentionally *not* the same as `contact_time_step(system, system)`: the latter
# would represent two identical copies of the same rigid body in pair contact and therefore
# uses the reduced mass `m/2` instead of the rigid-wall limit `m`.
@inline function contact_time_step(system::RigidBodySystem)
    return contact_time_step(system.contact_model, system)
end

@inline function contact_time_step(::Nothing, system::RigidBodySystem)
    return Inf
end

@inline function contact_time_step(contact_model::RigidContactModel,
                                   system::RigidBodySystem)
    # A wall is treated as an infinite-mass contact partner, so the reduced mass collapses
    # to the mass of the rigid body particle itself.
    return sqrt(minimum(system.mass) / contact_model.normal_stiffness)
end

@inline function contact_time_step(system::RigidBodySystem,
                                   neighbor::RigidBodySystem)
    if isinf(contact_time_step(system)) || isinf(contact_time_step(neighbor))
        return Inf
    end
    contact_model = system.contact_model::RigidContactModel
    neighbor_contact_model = neighbor.contact_model::RigidContactModel

    # For rigid-rigid contact, use one symmetric pair stiffness and the reduced mass of the
    # lightest contact-carrying particles of both bodies. This makes the estimate invariant
    # under swapping `system` and `neighbor`.
    pair_normal_stiffness = (contact_model.normal_stiffness +
                             neighbor_contact_model.normal_stiffness) / 2

    system_min_mass = minimum(system.mass)
    neighbor_min_mass = minimum(neighbor.mass)
    reduced_mass = system_min_mass * neighbor_min_mass /
                   (system_min_mass + neighbor_min_mass)

    return sqrt(reduced_mass / pair_normal_stiffness)
end

@inline function contact_time_step(system::RigidBodySystem,
                                   neighbor::WallBoundarySystem)
    # Wall boundaries do not carry their own rigid-body mass or inertia model, so the
    # wall-contact estimate is exactly the single-body rigid-wall limit.
    return contact_time_step(system)
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
    print(io, ")")
end

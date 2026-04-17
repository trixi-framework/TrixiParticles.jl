abstract type AbstractRigidContactModel end

"""
    RigidContactModel(; normal_stiffness,
                      normal_damping=0.0,
                      contact_distance=0.0)

Basic rigid contact model stored on a rigid body.
It is currently used for both rigid-wall and rigid-rigid contact.
The contact force consists of a linear normal spring-dashpot contribution only.
If `contact_distance == 0`, the particle spacing of the `RigidBodySystem` will be used as contact distance.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
struct RigidContactModel{ELTYPE <: Real} <: AbstractRigidContactModel
    normal_stiffness::ELTYPE
    normal_damping::ELTYPE
    contact_distance::ELTYPE
end

function RigidContactModel(; normal_stiffness,
                           normal_damping=0.0,
                           contact_distance=0.0)
    ELTYPE = promote_type(typeof(normal_stiffness),
                          typeof(normal_damping),
                          typeof(contact_distance))

    normal_stiffness_ = convert(ELTYPE, normal_stiffness)
    normal_damping_ = convert(ELTYPE, normal_damping)
    contact_distance_ = convert(ELTYPE, contact_distance)

    normal_stiffness_ > 0 ||
        throw(ArgumentError("`normal_stiffness` must be positive"))
    normal_damping_ >= 0 ||
        throw(ArgumentError("`normal_damping` must be non-negative"))
    contact_distance_ >= 0 ||
        throw(ArgumentError("`contact_distance` must be non-negative"))

    return RigidContactModel(normal_stiffness_, normal_damping_, contact_distance_)
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
                             contact_distance)
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
    print(io, ", contact_distance=", model.contact_distance)
    print(io, ")")
end

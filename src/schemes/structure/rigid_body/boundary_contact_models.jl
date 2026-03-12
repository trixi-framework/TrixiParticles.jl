abstract type AbstractRigidContactModel end

"""
    RigidContactModel(; normal_stiffness,
                      normal_damping=0.0,
                      contact_distance=0.0)

Basic rigid contact model stored on a rigid body.
In this branch it is used for rigid-wall contact only.
The contact force consists of a linear normal spring-dashpot contribution only.
If `contact_distance == 0`, `RigidBodySystem` uses its particle spacing.
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

function RigidContactModel(model::RigidContactModel, particle_spacing,
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

const RigidBoundaryContactModel = RigidContactModel

@inline function contact_time_step(system::RigidBodySystem)
    return contact_time_step(system.contact_model, system)
end

@inline function contact_time_step(::Nothing, system::RigidBodySystem)
    return Inf
end

function contact_time_step(contact_model::RigidContactModel,
                           system::RigidBodySystem)
    min_mass = minimum(system.mass)
    normal_stiffness = contact_model.normal_stiffness

    if min_mass <= eps(eltype(system)) || normal_stiffness <= eps(eltype(system))
        return Inf
    end

    return sqrt(min_mass / normal_stiffness)
end

function Base.show(io::IO, model::RigidContactModel)
    @nospecialize model # reduce precompilation time

    print(io, "RigidContactModel(")
    print(io, "normal_stiffness=", model.normal_stiffness)
    print(io, ", normal_damping=", model.normal_damping)
    print(io, ", contact_distance=", model.contact_distance)
    print(io, ")")
end

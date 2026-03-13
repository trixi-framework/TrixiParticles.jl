abstract type AbstractRigidContactModel end

"""
    RigidContactModel(; normal_stiffness,
                      normal_damping=0.0,
                      contact_distance=0.0)

Basic rigid contact model stored on a rigid body.
It is currently used for both rigid-wall and rigid-rigid contact.
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

@inline function rigid_contact_pair_parameters(contact_model::RigidContactModel,
                                               neighbor_contact_model::RigidContactModel,
                                               ::Type{ELTYPE}) where {ELTYPE}
    # Use a symmetric mixed-contact rule: average the linear spring-dashpot coefficients and
    # keep the larger contact shell so either body's cutoff can activate the pair.
    contact_distance = max(convert(ELTYPE, contact_model.contact_distance),
                           convert(ELTYPE, neighbor_contact_model.contact_distance))
    normal_stiffness = (convert(ELTYPE, contact_model.normal_stiffness) +
                        convert(ELTYPE, neighbor_contact_model.normal_stiffness)) / 2
    normal_damping = (convert(ELTYPE, contact_model.normal_damping) +
                      convert(ELTYPE, neighbor_contact_model.normal_damping)) / 2

    return (; contact_distance, normal_stiffness, normal_damping)
end

@inline function contact_time_step(system::RigidBodySystem)
    return contact_time_step(system.contact_model, system)
end

@inline function contact_time_step(system::RigidBodySystem, ::Nothing)
    return contact_time_step(system)
end

function contact_time_step(system::RigidBodySystem, semi)
    dt = contact_time_step(system)
    isnothing(system.contact_model) && return dt

    foreach_system(semi) do neighbor
        if neighbor isa RigidBodySystem && neighbor !== system
            dt = min(dt,
                     contact_time_step(system.contact_model, system,
                                       neighbor.contact_model, neighbor))
        end
    end

    return dt
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

@inline function contact_time_step(::RigidContactModel, system::RigidBodySystem,
                                   ::Nothing, neighbor_system::RigidBodySystem)
    return Inf
end

@inline function contact_time_step(::Nothing, system::RigidBodySystem,
                                   ::RigidContactModel,
                                   neighbor_system::RigidBodySystem)
    return Inf
end

function contact_time_step(contact_model::RigidContactModel,
                           system::RigidBodySystem,
                           neighbor_contact_model::RigidContactModel,
                           neighbor_system::RigidBodySystem)
    pair_parameters = rigid_contact_pair_parameters(contact_model, neighbor_contact_model,
                                                    eltype(system))
    normal_stiffness = pair_parameters.normal_stiffness
    min_mass = minimum(system.mass)
    neighbor_min_mass = minimum(neighbor_system.mass)

    if min_mass <= eps(eltype(system)) || neighbor_min_mass <= eps(eltype(system)) ||
       normal_stiffness <= eps(eltype(system))
        return Inf
    end

    reduced_mass = min_mass * neighbor_min_mass / (min_mass + neighbor_min_mass)
    return sqrt(reduced_mass / normal_stiffness)
end

function Base.show(io::IO, model::RigidContactModel)
    @nospecialize model # reduce precompilation time

    print(io, "RigidContactModel(")
    print(io, "normal_stiffness=", model.normal_stiffness)
    print(io, ", normal_damping=", model.normal_damping)
    print(io, ", contact_distance=", model.contact_distance)
    print(io, ")")
end

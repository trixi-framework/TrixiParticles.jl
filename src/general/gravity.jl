abstract type AbstractGravityModel end
abstract type AbstractGravitySoftening end

"""
    DEFAULT_GRAVITATIONAL_CONSTANT

Default unitless gravitational constant used by [`NewtonianGravity`](@ref).

TrixiParticles does not attach physical units to gravity models. Use `G = 1`
for nondimensional examples, or pass a numeric `gravitational_constant` in a
consistent unit system.
"""
const DEFAULT_GRAVITATIONAL_CONSTANT = 1.0

@doc raw"""
    NoSoftening()

Newtonian gravity without softening.

Calling `softening(distance)` returns the scalar force factor ``1 / r^3``.
Calling `softening(pos_diff)` or `softening(pos_diff, distance)` returns the
corresponding vector factor.
"""
struct NoSoftening <: AbstractGravitySoftening end

@doc raw"""
    PlummerSoftening(softening_length)
    PlummerSoftening(; softening_length)

Plummer gravitational softening with length ``\epsilon``.

Calling `softening(distance)` returns the scalar force factor
``1 / (r^2 + \epsilon^2)^{3/2}``. Calling `softening(pos_diff)` or
`softening(pos_diff, distance)` returns the corresponding vector factor.
"""
struct PlummerSoftening{ELTYPE <: Real} <: AbstractGravitySoftening
    softening_length::ELTYPE

    function PlummerSoftening(softening_length)
        if !isfinite(softening_length) || softening_length < zero(softening_length)
            throw(ArgumentError("`softening_length` must be non-negative and finite"))
        end

        return new{typeof(softening_length)}(softening_length)
    end
end

function PlummerSoftening(; softening_length)
    return PlummerSoftening(softening_length)
end

@doc raw"""
    NewtonianGravity(; gravitational_constant=DEFAULT_GRAVITATIONAL_CONSTANT,
                     softening=NoSoftening(), cutoff_radius=Inf)

Model for Newtonian pairwise self-gravity.

The default convention is unitless with ``G = 1``. No dependency on a unit package
is required; choose `gravitational_constant` consistently with the units used for
mass, length, and time in the simulation setup.

# Keywords
- `gravitational_constant=DEFAULT_GRAVITATIONAL_CONSTANT`: Strength of the pairwise
  gravity interaction.
- `softening=NoSoftening()`: Gravitational softening model.
- `cutoff_radius=Inf`: Maximum interaction distance used by pairwise interaction kernels.
"""
struct NewtonianGravity{ELTYPE <: Real, SOFTENING <: AbstractGravitySoftening,
                        CUTOFF} <: AbstractGravityModel
    gravitational_constant :: ELTYPE
    softening              :: SOFTENING
    cutoff_radius          :: ELTYPE

    function NewtonianGravity(; gravitational_constant=DEFAULT_GRAVITATIONAL_CONSTANT,
                              softening=NoSoftening(),
                              cutoff_radius=oftype(float(gravitational_constant), Inf))
        softening isa AbstractGravitySoftening ||
            throw(ArgumentError("`softening` must be a gravitational softening model"))

        gravitational_constant_, _,
        cutoff_radius_ = promote(gravitational_constant,
                                 softening_length_for_promotion(softening,
                                                                gravitational_constant),
                                 cutoff_radius)

        if !isfinite(gravitational_constant_) ||
           gravitational_constant_ < zero(gravitational_constant_)
            throw(ArgumentError("`gravitational_constant` must be non-negative and finite"))
        end

        if isnan(cutoff_radius_) || cutoff_radius_ <= zero(cutoff_radius_)
            throw(ArgumentError("`cutoff_radius` must be positive or `Inf`"))
        end

        softening_ = copy_softening_model(softening, typeof(gravitational_constant_))

        return new{typeof(gravitational_constant_), typeof(softening_),
                   !isinf(cutoff_radius_)}(gravitational_constant_,
                                           softening_,
                                           cutoff_radius_)
    end
end

@inline function (softening::AbstractGravitySoftening)(distance)
    return gravity_force_factor(softening, distance)
end

@inline function (softening::AbstractGravitySoftening)(pos_diff::AbstractVector)
    return softening(pos_diff, norm(pos_diff))
end

@inline function (softening::AbstractGravitySoftening)(pos_diff, distance)
    if iszero(distance)
        return zero(pos_diff)
    end

    return gravity_force_factor(softening, distance) * pos_diff
end

@inline softening_length_for_promotion(::NoSoftening,
                                       gravitational_constant) = zero(gravitational_constant)
@inline softening_length_for_promotion(softening,
                                       gravitational_constant) = softening.softening_length

@inline copy_softening_model(::NoSoftening, ::Type{ELTYPE}) where {ELTYPE} = NoSoftening()
@inline function copy_softening_model(softening::PlummerSoftening,
                                      ::Type{ELTYPE}) where {ELTYPE}
    return PlummerSoftening(convert(ELTYPE, softening.softening_length))
end

@inline function gravity_force_factor(::NoSoftening, distance)
    return inv(distance^3)
end

@inline function gravity_force_factor(softening::PlummerSoftening, distance)
    (; softening_length) = softening

    distance_square = distance^2 + softening_length^2

    return inv(distance_square * sqrt(distance_square))
end

@inline function gravity_potential_factor(::NoSoftening, distance)
    return inv(distance)
end

@inline function gravity_potential_factor(softening::PlummerSoftening, distance)
    (; softening_length) = softening

    return inv(sqrt(distance^2 + softening_length^2))
end

"""
    gravitational_mass(system, particle)

Return the mass used by gravity interactions for `particle`.
"""
@propagate_inbounds function gravitational_mass(system, particle)
    return system.mass[particle]
end

"""
    current_position(u, system, particle)

Return the current particle position used by gravity interactions.
"""
@propagate_inbounds function current_position(u, system, particle)
    return current_coords(u, system, particle)
end

@inline function current_position(u, system)
    return current_coordinates(u, system)
end

@inline gravity_model(system) = nothing

@inline function gravity_model(particle_system, neighbor_system)
    return gravity_model(particle_system)
end

@inline function gravity_acceleration(gravity::NewtonianGravity, pos_diff, distance,
                                      neighbor_mass)
    if iszero(distance)
        return zero(pos_diff)
    end

    return gravity_acceleration_factor(gravity, distance, neighbor_mass) * pos_diff
end

@inline function gravity_acceleration_factor(gravity::NewtonianGravity{ELTYPE, SOFTENING,
                                                                       false},
                                             distance,
                                             neighbor_mass) where {ELTYPE, SOFTENING}
    (; gravitational_constant, softening) = gravity

    return -gravitational_constant * neighbor_mass *
           gravity_force_factor(softening, distance)
end

@inline function gravity_acceleration_factor(gravity::NewtonianGravity{ELTYPE, SOFTENING,
                                                                       true},
                                             distance,
                                             neighbor_mass) where {ELTYPE, SOFTENING}
    (; gravitational_constant, softening, cutoff_radius) = gravity

    distance > cutoff_radius && return zero(distance)

    return -gravitational_constant * neighbor_mass *
           gravity_force_factor(softening, distance)
end

@inline function gravity_acceleration(::Nothing, pos_diff, distance, neighbor_mass)
    return zero(pos_diff)
end

"""
    gravity_acceleration!(dv, gravity, particle_system, neighbor_system,
                          particle, neighbor, pos_diff, distance)

Accumulate the acceleration induced by `neighbor` on `particle` in `dv`.
"""
@inline function gravity_acceleration!(dv, gravity::NewtonianGravity,
                                       particle_system, neighbor_system,
                                       particle, neighbor, pos_diff, distance)
    iszero(distance) && return dv

    factor = gravity_acceleration_factor(gravity, distance,
                                         gravitational_mass(neighbor_system, neighbor))

    @inbounds for i in 1:ndims(particle_system)
        dv[i, particle] += factor * pos_diff[i]
    end

    return dv
end

@inline function gravity_acceleration!(dv, ::Nothing,
                                       particle_system, neighbor_system,
                                       particle, neighbor, pos_diff, distance)
    return dv
end

@inline function gravity_interaction!(dv_particle, gravity::NewtonianGravity,
                                      particle_system, neighbor_system, particle,
                                      neighbor, pos_diff, distance, m_a, m_b)
    dv_particle[] += gravity_acceleration(gravity, pos_diff, distance, m_b)

    return dv_particle
end

@inline function gravity_interaction!(dv_particle, ::Nothing, particle_system,
                                      neighbor_system, particle, neighbor,
                                      pos_diff, distance, m_a, m_b)
    return dv_particle
end

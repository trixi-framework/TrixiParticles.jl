abstract type AbstractGravityModel end

@doc raw"""
    NewtonianGravity(; gravitational_constant, softening_length=0, cutoff_radius=Inf)

Model for Newtonian pairwise self-gravity.

# Keywords
- `gravitational_constant`: Strength of the pairwise gravity interaction.
- `softening_length=0`: Distance regularization used by pairwise interaction kernels.
- `cutoff_radius=Inf`: Maximum interaction distance used by pairwise interaction kernels.
"""
struct NewtonianGravity{ELTYPE <: Real} <: AbstractGravityModel
    gravitational_constant :: ELTYPE
    softening_length       :: ELTYPE
    cutoff_radius          :: ELTYPE

    function NewtonianGravity(; gravitational_constant,
                              softening_length=zero(gravitational_constant),
                              cutoff_radius=oftype(float(gravitational_constant), Inf))
        gravitational_constant_, softening_length_, cutoff_radius_ = promote(gravitational_constant,
                                                                             softening_length,
                                                                             cutoff_radius)

        if gravitational_constant_ < zero(gravitational_constant_)
            throw(ArgumentError("`gravitational_constant` must be non-negative"))
        end

        if softening_length_ < zero(softening_length_)
            throw(ArgumentError("`softening_length` must be non-negative"))
        end

        if cutoff_radius_ <= zero(cutoff_radius_)
            throw(ArgumentError("`cutoff_radius` must be positive"))
        end

        return new{typeof(gravitational_constant_)}(gravitational_constant_,
                                                    softening_length_,
                                                    cutoff_radius_)
    end
end

@inline gravity_model(system) = nothing

@inline function gravity_model(particle_system, neighbor_system)
    return gravity_model(particle_system)
end

@inline function gravity_acceleration(gravity::NewtonianGravity, pos_diff, distance,
                                      neighbor_mass)
    (; gravitational_constant, softening_length, cutoff_radius) = gravity

    if distance > cutoff_radius || iszero(distance)
        return zero(pos_diff)
    end

    distance_square = distance^2 + softening_length^2
    inverse_distance_cube = inv(distance_square * sqrt(distance_square))

    return -gravitational_constant * neighbor_mass * inverse_distance_cube * pos_diff
end

@inline function gravity_acceleration(::Nothing, pos_diff, distance, neighbor_mass)
    return zero(pos_diff)
end

@inline function gravity_interaction!(dv_particle, gravity, particle_system,
                                      neighbor_system, particle, neighbor,
                                      pos_diff, distance, m_a, m_b)
    return dv_particle
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

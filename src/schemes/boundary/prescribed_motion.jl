"""
    PrescribedMotion(movement_function, is_moving; moving_particles=nothing)

# Arguments
- `movement_function`: Function of `(x, t)` where `x` is an `SVector` of the *initial*
                       particle position and `t` is the time, returning an `SVector`
                       of ``d`` dimensions for a ``d``-dimensional problem containing
                       the new particle position at time `t`.
- `is_moving`:         Function of `t` to determine in each timestep if the particles
                       are moving or not. Its boolean return value determines
                       if the neighborhood search will be updated.

# Keyword Arguments
- `moving_particles`: Indices of moving particles. Default is each particle in the system
                      for the [`WallBoundarySystem`](@ref) and all clamped particles
                      for the [`TotalLagrangianSPHSystem`](@ref).

# Examples
```jldoctest; output = false
# Circular motion of particles for t < 1.5
movement_function(x, t) = x + SVector(cos(2pi * t), sin(2pi * t))
is_moving(t) = t < 1.5

motion = PrescribedMotion(movement_function, is_moving)

# Rotation around the origin
movement_function2(x, t) = SVector(cos(2pi * t) * x[1] - sin(2pi * t) * x[2],
                                   sin(2pi * t) * x[1] + cos(2pi * t) * x[2])
is_moving2(t) = true

motion2 = PrescribedMotion(movement_function2, is_moving2)

# output
PrescribedMotion{typeof(movement_function2), typeof(is_moving2), Vector{Int64}}(movement_function2, is_moving2, Int64[])
```
"""
struct PrescribedMotion{MF, IM, MP}
    movement_function :: MF
    is_moving         :: IM
    moving_particles  :: MP # Vector{Int}
end

# The default constructor needs to be accessible for Adapt.jl to work with this struct.
# See the comments in general/gpu.jl for more details.
function PrescribedMotion(movement_function, is_moving; moving_particles=nothing)
    # Default value is an empty vector, which will be resized in the `WallBoundarySystem`
    # constructor to move all particles.
    moving_particles = isnothing(moving_particles) ? Int[] : vec(moving_particles)

    return PrescribedMotion(movement_function, is_moving, moving_particles)
end

function initialize_prescribed_motion!(prescribed_motion::PrescribedMotion,
                                       initial_condition,
                                       n_clamped_particles=nparticles(initial_condition))
    # Test `movement_function` return type
    pos = extract_svector(initial_condition.coordinates,
                          Val(size(initial_condition.coordinates, 1)), 1)
    if !(prescribed_motion.movement_function(pos, 0.0) isa SVector)
        @warn "Return value of `movement_function` is not of type `SVector`. " *
              "Returning regular `Vector`s causes allocations and significant performance overhead."
    end

    # Empty `moving_particles` means all clamped particles are moving.
    # For boundaries, all particles are considered clamped.
    if isempty(prescribed_motion.moving_particles)
        # Default is an empty vector, since the number of particles is not known when
        # instantiating `PrescribedMotion`.
        resize!(prescribed_motion.moving_particles, n_clamped_particles)

        # Clamped particles for TLSPH are the last `n_clamped_particles` in the system
        first_particle = nparticles(initial_condition) - n_clamped_particles + 1
        clamped_particles = first_particle:nparticles(initial_condition)
        prescribed_motion.moving_particles .= collect(clamped_particles)
    end

    return prescribed_motion
end

function initialize_prescribed_motion!(::Nothing, initial_condition,
                                       n_clamped_particles=nparticles(initial_condition))
    return nothing
end

function (prescribed_motion::PrescribedMotion)(coordinates, velocity, acceleration,
                                               ismoving, system, semi, t)
    (; movement_function, is_moving, moving_particles) = prescribed_motion

    ismoving[] = is_moving(t)

    is_moving(t) || return nothing

    @threaded semi for particle in moving_particles
        pos_original = initial_coords(system, particle)
        pos_new = movement_function(pos_original, t)
        pos_deriv(t_) = ForwardDiff.derivative(t__ -> movement_function(pos_original, t__),
                                               t_)
        vel = pos_deriv(t)
        acc = ForwardDiff.derivative(pos_deriv, t)

        @inbounds for i in 1:ndims(system)
            coordinates[i, particle] = pos_new[i]
            velocity[i, particle] = vel[i]
            acceleration[i, particle] = acc[i]
        end
    end

    return nothing
end

"""
    OscillatingMotion2D(; frequency, translation_vector, rotation_angle, rotation_center,
                        rotation_phase_offset=0, tspan=(-Inf, Inf),
                        ramp_up_tspan=(0.0, 0.0), moving_particles=nothing)

Create a [`PrescribedMotion`](@ref) for a 2D oscillating motion of particles.
The motion is a combination of a translation and a rotation around a center point
with the same frequency. Note that a phase offset can be specified for a rotation
that is out of sync with the translation.

Create a [`PrescribedMotion`](@ref) for 2D oscillatory particle motion
that combines a translation and a rotation about a specified center.
Both use the same frequency; an optional phase offset allows the rotation
to be out of phase with the translation.

# Keywords
- `frequency`:          Frequency of the oscillation in Hz.
- `translation_vector`: Vector specifying the amplitude and direction of the translation.
- `rotation_angle`:     Maximum rotation angle in radians.
- `rotation_center`:    Center point of the rotation as an `SVector`.

# Keywords (optional)
- `rotation_phase_offset=0`: Phase offset of the rotation in number of periods (`1` = 1 period).
- `tspan=(-Inf, Inf)`:  Time span in which the motion is active. Outside of this time span,
                        particles remain at their last position.
- `ramp_up_tspan`:      Time span in which the motion is smoothly ramped up from zero
                        to full amplitude.
                        Outside of this time span, the motion has full amplitude.
                        Default is no ramp-up.
- `moving_particles`:   Indices of moving particles. Default is each particle in the system
                        for the [`WallBoundarySystem`](@ref) and all clamped particles
                        for the [`TotalLagrangianSPHSystem`](@ref).

# Examples
```jldoctest; output = false, filter = r"PrescribedMotion{.*"
# Oscillating motion with frequency 1 Hz, translation amplitude 1 in x-direction,
# rotation angle 90 degrees (pi/2) around the origin.
frequency = 1.0
translation_vector = SVector(1.0, 0.0)
rotation_angle = pi / 2
rotation_center = SVector(0.0, 0.0)

motion = OscillatingMotion2D(frequency, translation_vector, rotation_angle,
                             rotation_center)

# output
PrescribedMotion{...}
"""
function OscillatingMotion2D(; frequency, translation_vector, rotation_angle,
                             rotation_center, rotation_phase_offset=0, tspan=(-Inf, Inf),
                             ramp_up_tspan=(0.0, 0.0), moving_particles=nothing)
    @inline function movement_function(x, t)
        if isfinite(tspan[1])
            t = t - tspan[1]
        end

        sin_scaled = sin(frequency * 2pi * t)
        translation = sin_scaled * translation_vector
        x_centered = x .- rotation_center
        sin_scaled_offset = sin(2pi * (frequency * t - rotation_phase_offset))
        angle = rotation_angle * sin_scaled_offset
        rotated = SVector(x_centered[1] * cos(angle) - x_centered[2] * sin(angle),
                          x_centered[1] * sin(angle) + x_centered[2] * cos(angle))

        result = rotated .+ rotation_center .+ translation

        if ramp_up_tspan[2] > ramp_up_tspan[1] && ramp_up_tspan[1] <= t <= ramp_up_tspan[2]
            # Apply a smoothstep ramp-up during the ramp-up time span
            t_rel = (t - ramp_up_tspan[1]) / (ramp_up_tspan[2] - ramp_up_tspan[1])
            ramp_factor = 3 * t_rel^2 - 2 * t_rel^3
            return result * ramp_factor + (1 - ramp_factor) * x
        end
        return result
    end

    is_moving(t) = tspan[1] <= t <= tspan[2]

    return PrescribedMotion(movement_function, is_moving; moving_particles)
end

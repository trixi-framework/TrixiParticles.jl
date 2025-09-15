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
- `moving_particles`: Indices of moving particles. Default is each particle in the system.

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

function initialize!(prescribed_motion::PrescribedMotion, initial_condition)
    # Test `movement_function` return type`
    pos = extract_svector(initial_condition.coordinates,
                          Val(size(initial_condition.coordinates, 1)), 1)
    if !(prescribed_motion.movement_function(pos, 0.0) isa SVector)
        @warn "Return value of `movement_function` is not of type `SVector`. " *
              "Returning regular `Vector`s causes allocations and significant performance overhead."
    end

    # Empty `moving_particles` means all particles are moving
    if isempty(prescribed_motion.moving_particles)
        # Default is an empty vector, since the number of particles is not known when
        # instantiating `PrescribedMotion`.
        resize!(prescribed_motion.moving_particles, nparticles(initial_condition))
        prescribed_motion.moving_particles .= collect(1:nparticles(initial_condition))
    end
end

function (prescribed_motion::PrescribedMotion)(system, t, semi)
    (; coordinates, cache) = system
    (; movement_function, is_moving, moving_particles) = prescribed_motion
    (; acceleration, velocity) = cache

    system.ismoving[] = is_moving(t)

    is_moving(t) || return system

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

    return system
end

function (prescribed_motion::Nothing)(system::AbstractSystem, t, semi)
    system.ismoving[] = false

    return system
end

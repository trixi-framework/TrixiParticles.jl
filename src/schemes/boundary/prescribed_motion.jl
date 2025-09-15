"""
    PrescribedMotion(movement_function, is_moving; moving_particles=nothing)

# Arguments
- `movement_function`: Time-dependent function returning an `SVector` of ``d`` dimensions
                       for a ``d``-dimensional problem.
- `is_moving`: Function to determine in each timestep if the particles are moving or not. Its
   boolean return value is mandatory to determine if the neighborhood search will be updated.

# Keyword Arguments
- `moving_particles`: Indices of moving particles. Default is each particle in the system.

# Examples
In the example below, `motion` describes particles moving in a circle as long as
the time is lower than `1.5`.

```jldoctest; output = false
movement_function(t) = SVector(cos(2pi*t), sin(2pi*t))
is_moving(t) = t < 1.5

motion = PrescribedMotion(movement_function, is_moving)

# output
PrescribedMotion{typeof(movement_function), typeof(is_moving), Vector{Int64}}(movement_function, is_moving, Int64[])
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
    if !(movement_function(0.0) isa SVector)
        @warn "Return value of `movement_function` is not of type `SVector`. " *
              "Returning regular `Vector`s causes allocations and significant performance overhead."
    end

    # Default value is an empty vector, which will be resized in the `WallBoundarySystem`
    # constructor to move all particles.
    moving_particles = isnothing(moving_particles) ? Int[] : vec(moving_particles)

    return PrescribedMotion(movement_function, is_moving, moving_particles)
end

function (prescribed_motion::PrescribedMotion)(system, t, semi)
    (; coordinates, cache) = system
    (; movement_function, is_moving, moving_particles) = prescribed_motion
    (; acceleration, velocity) = cache

    system.ismoving[] = is_moving(t)

    is_moving(t) || return system

    @threaded semi for particle in moving_particles
        pos_new = initial_coords(system, particle) + movement_function(t)
        vel = ForwardDiff.derivative(movement_function, t)
        acc = ForwardDiff.derivative(t_ -> ForwardDiff.derivative(movement_function, t_), t)

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

"""
    BoundarySPHSystem(initial_condition, boundary_model; movement=nothing)

System for boundaries modeled by boundary particles.
The interaction between fluid and boundary particles is specified by the boundary model.

# Arguments
- `initial_condition`: Initial condition (see [`InitialCondition`](@ref))
- `boundary_model`: Boundary model (see [Boundary Models](@ref boundary_models))

# Keyword Arguments
- `movement`: For moving boundaries, a [`BoundaryMovement`](@ref) can be passed.
"""
struct BoundarySPHSystem{BM, NDIMS, ELTYPE <: Real, M, C} <: BoundarySystem{NDIMS}
    initial_condition :: InitialCondition{ELTYPE}
    coordinates       :: Array{ELTYPE, 2}
    boundary_model    :: BM
    movement          :: M
    ismoving          :: Vector{Bool}
    cache             :: C

    function BoundarySPHSystem(initial_condition, model; movement=nothing)
        coordinates = copy(initial_condition.coordinates)
        NDIMS = size(coordinates, 1)
        ismoving = zeros(Bool, 1)

        cache = create_cache_boundary(movement, initial_condition)

        if movement !== nothing && isempty(movement.moving_particles)
            # Default is an empty vector, since the number of particles is not known when
            # instantiating `BoundaryMovement`.
            resize!(movement.moving_particles, nparticles(initial_condition))
            movement.moving_particles .= collect(1:nparticles(initial_condition))
        end

        return new{typeof(model), NDIMS, eltype(coordinates), typeof(movement),
                   typeof(cache)}(initial_condition, coordinates, model, movement,
                                  ismoving, cache)
    end
end

"""
    BoundaryMovement(movement_function, is_moving; moving_particles=nothing)

# Arguments
- `movement_function`: Time-dependent function returning an `SVector` of ``d`` dimensions
                       for a ``d``-dimensional problem.
- `is_moving`: Function to determine in each timestep if the particles are moving or not. Its
    boolean return value is mandatory to determine if the neighborhood search will be updated.

# Keyword Arguments
- `moving_particles`: Indices of moving particles. Default is each particle in [`BoundarySPHSystem`](@ref).

In the example below, `movement` describes particles moving in a circle as long as
the time is lower than `1.5`.

# Examples
```julia
movement_function(t) = SVector(cos(2pi*t), sin(2pi*t))
is_moving(t) = t < 1.5

movement = BoundaryMovement(movement_function, is_moving)
```
"""
struct BoundaryMovement{MF, IM}
    movement_function :: MF
    is_moving         :: IM
    moving_particles  :: Vector{Int}

    function BoundaryMovement(movement_function, is_moving; moving_particles=nothing)
        if !(movement_function(0.0) isa SVector)
            @warn "Return value of `movement_function` is not of type `SVector`. " *
                  "Returning regular `Vector`s causes allocations and significant performance overhead."
        end

        # Default value is an empty vector, which will be resized in the `BoundarySPHSystem`
        # constructor to move all particles.
        moving_particles = isnothing(moving_particles) ? [] : vec(moving_particles)

        return new{typeof(movement_function),
                   typeof(is_moving)}(movement_function, is_moving, moving_particles)
    end
end

create_cache_boundary(::Nothing, initial_condition) = (;)

function create_cache_boundary(::BoundaryMovement, initial_condition)
    velocity = zero(initial_condition.velocity)
    acceleration = zero(initial_condition.velocity)
    return (; velocity, acceleration)
end

function Base.show(io::IO, system::BoundarySPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "BoundarySPHSystem{", ndims(system), "}(")
    print(io, system.boundary_model)
    print(io, ", ", system.movement)
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::BoundarySPHSystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "BoundarySPHSystem{$(ndims(system))}")
        summary_line(io, "#particles", nparticles(system))
        summary_line(io, "boundary model", system.boundary_model)
        summary_line(io, "movement function",
                     isnothing(system.movement) ? "nothing" :
                     string(system.movement.movement_function))
        summary_footer(io)
    end
end

timer_name(::BoundarySPHSystem) = "boundary"

function (movement::BoundaryMovement)(system, t)
    (; coordinates, cache) = system
    (; movement_function, is_moving, moving_particles) = movement
    (; acceleration, velocity) = cache

    system.ismoving[1] = is_moving(t)

    is_moving(t) || return system

    @threaded for particle in moving_particles
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

function (movement::Nothing)(system, t)
    system.ismoving[1] = false

    return system
end

@inline function nparticles(system::BoundarySPHSystem)
    length(system.boundary_model.hydrodynamic_mass)
end

# No particle positions are advanced for boundary systems,
# except when using `BoundaryModelDummyParticles` with `ContinuityDensity`.
@inline function n_moving_particles(system::BoundarySPHSystem)
    return 0
end

@inline function n_moving_particles(system::BoundarySPHSystem{<:BoundaryModelDummyParticles{ContinuityDensity}})
    return nparticles(system)
end

@inline u_nvariables(system::BoundarySPHSystem) = 0

# For BoundaryModelDummyParticles with ContinuityDensity, this needs to be 1.
# For all other models and density calculators, it's irrelevant.
@inline v_nvariables(system::BoundarySPHSystem) = 1

@inline function current_coordinates(u, system::BoundarySPHSystem)
    return system.coordinates
end

@inline function current_velocity(v, system::BoundarySPHSystem, particle)
    (; cache, ismoving) = system

    if ismoving[1]
        return extract_svector(cache.velocity, system, particle)
    end

    return SVector(ntuple(_ -> 0.0, Val(ndims(system))))
end

@inline function current_acceleration(system::BoundarySPHSystem, particle)
    (; cache, ismoving) = system

    if ismoving[1]
        return extract_svector(cache.acceleration, system, particle)
    end

    return SVector(ntuple(_ -> 0.0, Val(ndims(system))))
end

@inline function viscous_velocity(v, system::BoundarySPHSystem, particle)
    return viscous_velocity(v, system.boundary_model.viscosity, system, particle)
end

@inline function viscous_velocity(v, ::ViscosityAdami, system, particle)
    return extract_svector(system.boundary_model.cache.wall_velocity, system, particle)
end

@inline function viscous_velocity(v, viscosity, system, particle)
    return current_velocity(v, system, particle)
end

@inline function particle_density(v, system::BoundarySPHSystem, particle)
    return particle_density(v, system.boundary_model, system, particle)
end

@inline function particle_pressure(v, system::BoundarySPHSystem, particle)
    return particle_pressure(v, system.boundary_model, system, particle)
end

@inline function hydrodynamic_mass(system::BoundarySPHSystem, particle)
    return system.boundary_model.hydrodynamic_mass[particle]
end

@inline function smoothing_kernel(system::BoundarySPHSystem, distance)
    (; smoothing_kernel, smoothing_length) = system.boundary_model
    return kernel(smoothing_kernel, distance, smoothing_length)
end

function update_positions!(system::BoundarySPHSystem, v, u, v_ode, u_ode, semi, t)
    (; movement) = system

    movement(system, t)
end

function update_quantities!(system::BoundarySPHSystem, v, u, v_ode, u_ode, semi, t)
    (; boundary_model) = system

    update_density!(boundary_model, system, v, u, v_ode, u_ode, semi)

    return system
end

# This update depends on the computed quantities of the fluid system and therefore
# has to be in `update_final!` after `update_quantities!`.
function update_final!(system::BoundarySPHSystem, v, u, v_ode, u_ode, semi, t)
    (; boundary_model) = system

    update_pressure!(boundary_model, system, v, u, v_ode, u_ode, semi)

    return system
end

function write_u0!(u0, system::BoundarySPHSystem)
    return u0
end

function write_v0!(v0, system::BoundarySPHSystem)
    return v0
end

function write_v0!(v0,
                   system::BoundarySPHSystem{<:BoundaryModelDummyParticles{ContinuityDensity}})
    (; cache) = system.boundary_model
    (; initial_density) = cache

    for particle in eachparticle(system)
        # Set particle densities
        v0[1, particle] = initial_density[particle]
    end

    return v0
end

function restart_with!(system::BoundarySPHSystem, v, u)
    return system
end

function restart_with!(system::BoundarySPHSystem{<:BoundaryModelDummyParticles{ContinuityDensity}},
                       v, u)
    (; initial_density) = model.cache

    for particle in eachparticle(system)
        initial_density[particle] = v[1, particle]
    end

    return system
end

function viscosity_model(system::BoundarySPHSystem)
    return system.boundary_model.viscosity
end

function calculate_dt(v_ode, u_ode, cfl_number, system::BoundarySystem)
    return Inf
end

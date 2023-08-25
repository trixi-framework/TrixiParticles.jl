"""
    BoundarySPHSystem(inititial_condition, model; movement=nothing)

System for boundaries modeled by boundary particles.
The interaction between fluid and boundary particles is specified by the boundary model.

For moving boundaries, a [`BoundaryMovement`](@ref)) can be passed with the keyword
argument `movement`.
"""
struct BoundarySPHSystem{BM, NDIMS, ELTYPE <: Real, M, C} <: System{NDIMS}
    coordinates    :: Array{ELTYPE, 2}
    boundary_model :: BM
    movement       :: M
    ismoving       :: Vector{Bool}
    cache          :: C

    function BoundarySPHSystem(inititial_condition, model; movement=nothing)
        coordinates = inititial_condition.coordinates
        NDIMS = size(coordinates, 1)
        ismoving = zeros(Bool, 1)

        cache = create_cache(movement, inititial_condition)

        return new{typeof(model), NDIMS, eltype(coordinates), typeof(movement),
                   typeof(cache)}(coordinates, model, movement,
                                  ismoving, cache)
    end
end

"""
    BoundaryMovement(movement_function, is_moving)

# Arguments
- `movement_function`: Tuple containing a time dependent function in each dimension
- `is_moving`: Function to determine in each timestep if the particles are moving or not. Its
    boolean return value is mandatory to determine if the neighborhood search will be updated.


In the example below, `movement` describes particles moving in a circle as long as
the time is lower than `1.5`.

# Examples
```julia
f_x(t) = cos(2pi*t)
f_y(t) = sin(2pi*t)
is_moving(t) = t < 1.5

movement = BoundaryMovement((f_x, f_y), is_moving)
```
"""
struct BoundaryMovement{MF, IM}
    movement_function :: MF
    is_moving         :: IM

    function BoundaryMovement(movement_function, is_moving)
        return new{typeof(movement_function), typeof(is_moving)}(movement_function,
                                                                 is_moving)
    end
end

create_cache(::Nothing, inititial_condition) = (;)

function create_cache(::BoundaryMovement, inititial_condition)
    initial_coordinates = copy(inititial_condition.coordinates)
    velocity = similar(inititial_condition.velocity)
    acceleration = similar(inititial_condition.velocity)
    return (; initial_coordinates, velocity, acceleration)
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
        summary_line(io, "movement function", system.movement)
        summary_footer(io)
    end
end

timer_name(::BoundarySPHSystem) = "boundary"

@inline Base.eltype(system::BoundarySPHSystem) = eltype(system.coordinates)

# This does not account for moving boundaries, but it's only used to initialize the
# neighborhood search, anyway.
@inline initial_coordinates(system::BoundarySPHSystem) = system.coordinates

function (movement::BoundaryMovement)(system, t)
    (; coordinates, cache) = system
    (; movement_function, is_moving) = movement
    (; acceleration, velocity, initial_coordinates) = cache

    system.ismoving[1] = is_moving(t)

    is_moving(t) || return system

    for particle in eachparticle(system)
        for i in 1:ndims(system)
            coordinates[i, particle] = movement_function[i](t) +
                                       initial_coordinates[i, particle]

            velocity[i, particle] = ForwardDiff.derivative(movement_function[i], t)
            acceleration[i, particle] = ForwardDiff.derivative(t_ -> ForwardDiff.derivative(movement_function[i],
                                                                                            t_),
                                                               t)
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
# except when using BoundaryModelDummyParticles with ContinuityDensity.
@inline function n_moving_particles(system::BoundarySPHSystem)
    return 0
end

@inline function n_moving_particles(system::BoundarySPHSystem{
                                                              <:BoundaryModelDummyParticles
                                                              })
    return n_moving_particles(system, system.boundary_model.density_calculator)
end

@inline function n_moving_particles(system::BoundarySPHSystem, density_calculator)
    return 0
end

@inline function n_moving_particles(system::BoundarySPHSystem, ::ContinuityDensity)
    nparticles(system)
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
    return extract_svector(system.boundary_model.cache.wall_velocity, system, particle)
end

@inline function particle_density(v, system::BoundarySPHSystem, particle)
    return particle_density(v, system.boundary_model, system, particle)
end

@inline function hydrodynamic_mass(system::BoundarySPHSystem, particle)
    return system.boundary_model.hydrodynamic_mass[particle]
end

@inline function smoothing_kernel(system::BoundarySPHSystem, distance)
    (; smoothing_kernel, smoothing_length) = system.boundary_model
    return kernel(smoothing_kernel, distance, smoothing_length)
end

function update_positions!(system::BoundarySPHSystem, system_index, v, u, v_ode, u_ode,
                           semi, t)
    (; movement) = system

    movement(system, t)
end

function update_quantities!(system::BoundarySPHSystem, system_index, v, u, v_ode, u_ode,
                            semi, t)
    (; boundary_model) = system

    update_density!(boundary_model, system, system_index, v, u, v_ode, u_ode, semi)

    return system
end

# This update depends on the computed quantities of the fluid system and therefore
# has to be in `update_final!` after `update_quantities!`.
function update_final!(system::BoundarySPHSystem, system_index, v, u, v_ode, u_ode, semi, t)
    (; boundary_model) = system

    update_pressure!(boundary_model, system, system_index, v, u, v_ode, u_ode, semi)

    return system
end

function write_u0!(u0, system::BoundarySPHSystem)
    return u0
end

function write_v0!(v0, system::BoundarySPHSystem{<:BoundaryModelMonaghanKajtar})
    return v0
end

function write_v0!(v0, system::BoundarySPHSystem{<:BoundaryModelDummyParticles})
    (; density_calculator) = system.boundary_model

    write_v0!(v0, density_calculator, system)
end

function write_v0!(v0, density_calculator, system::BoundarySPHSystem)
    return v0
end

function write_v0!(v0, ::ContinuityDensity, system::BoundarySPHSystem)
    (; cache) = system.boundary_model
    (; initial_density) = cache

    for particle in eachparticle(system)
        # Set particle densities
        v0[1, particle] = initial_density[particle]
    end

    return v0
end

function restart_with!(system::BoundarySPHSystem, v, u)
    restart_with!(system, system.boundary_model, v, u)
end

function restart_with!(system, ::BoundaryModelMonaghanKajtar, v, u)
    return system
end

function restart_with!(system, model::BoundaryModelDummyParticles, v, u)
    restart_with!(system, model, model.density_calculator, v, u)
end

function restart_with!(system, model, density_calculator, v, u)
    return system
end

function restart_with!(system, model, ::ContinuityDensity, v, u)
    (; initial_density) = model.cache

    for particle in eachparticle(system)
        initial_density[particle] = v[1, particle]
    end

    return system
end

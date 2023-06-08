"""
    BoundarySPHSystem(coordinates, model; move_coordinates=nothing)

System for boundaries modeled by boundary particles.
The system is initialized with the coordinates of the particles and their masses.
The interaction between fluid and boundary particles is specified by the boundary model.

The `move_coordinates` function is to define in which way the boundary particles move over time.
    (See [`MovementFunction`](@ref))
"""
struct BoundarySPHSystem{BM, NDIMS, ELTYPE <: Real, MF} <: System{NDIMS}
    coordinates       :: Array{ELTYPE, 2}
    boundary_model    :: BM
    move_coordinates! :: MF
    ismoving          :: Vector{Bool}

    function BoundarySPHSystem(coordinates, model; move_coordinates=nothing)
        NDIMS = size(coordinates, 1)
        ismoving = zeros(Bool, 1)

        return new{typeof(model), NDIMS,
                   eltype(coordinates),
                   typeof(move_coordinates)}(coordinates, model, move_coordinates, ismoving)
    end
end

function Base.show(io::IO, system::BoundarySPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "BoundarySPHSystem{", ndims(system), "}(")
    print(io, system.boundary_model)
    print(io, ", ", system.move_coordinates!)
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
        summary_line(io, "movement function", system.move_coordinates!)
        summary_footer(io)
    end
end

@inline Base.eltype(system::BoundarySPHSystem) = eltype(system.coordinates)

# This does not account for moving boundaries, but it's only used to initialize the
# neighborhood search, anyway.
@inline initial_coordinates(system::BoundarySPHSystem) = system.coordinates

"""
    MovementFunction(movement_function, coordinates, keep_moving)

# Arguments
- `movement_function`: Tuple containing a time dependent function in each dimension
- `coordinates`: Coordinates of the initial positions
- `keep_moving`: Function to determine in each timestep if the particles are moving or not. Its
    boolean return value is mandatory to determines if the neighborhood search will be updated.


In the example below `move_coordinates` describes two particles moving in a circle as long as
the time is lower than `1.5`.

# Examples
```julia
f_x(t) = cos(2pi*t)
f_y(t) = sin(2pi*t)
keep_moving(t) = t < 1.5

move_coordinates = MovementFunction((f_x, f_y), [0.0 1.0; 1.0 1.0], keep_moving)
```
"""
struct MovementFunction{NDIMS, ELTYPE <: Real, MF, KM}
    initial_position  :: Array{ELTYPE, 2}
    velocity          :: Array{ELTYPE, 2}
    acceleration      :: Array{ELTYPE, 2}
    movement_function :: MF
    keep_moving       :: KM

    function MovementFunction(movement_function, coordinates, keep_moving)
        NDIMS = size(coordinates, 1)
        velocity = zeros(size(coordinates))
        acceleration = zeros(size(coordinates))
        initial_position = copy(coordinates)

        return new{NDIMS, eltype(initial_position), typeof(movement_function),
                   typeof(keep_moving)}(initial_position, velocity, acceleration,
                                        movement_function, keep_moving)
    end
end

function (move_coordinates!::MovementFunction)(system, t)
    @unpack coordinates = system
    @unpack initial_position, movement_function, keep_moving,
    velocity, acceleration = move_coordinates!

    system.ismoving[1] = keep_moving(t)

    if keep_moving(t)
        for particle in eachparticle(system)
            for i in 1:ndims(system)
                coordinates[i, particle] = movement_function[i](t) +
                                           initial_position[i, particle]

                velocity[i, particle] = ForwardDiff.derivative(movement_function[i], t)
                acceleration[i, particle] = ForwardDiff.derivative(t_ -> ForwardDiff.derivative(movement_function[i],
                                                                                                t_),
                                                                   t)
            end
        end
    end

    return system
end

function (move_coordinates!::Nothing)(system, t)
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
    @unpack move_coordinates!, ismoving = system

    ismoving[1] && (return extract_svector(move_coordinates!.velocity, system, particle))

    return SVector(ntuple(_ -> 0.0, Val(ndims(system))))
end

@inline function current_acceleration(system, particle)
    return SVector(ntuple(_ -> 0.0, Val(ndims(system))))
end

@inline function current_acceleration(system::BoundarySPHSystem, particle)
    @unpack move_coordinates!, ismoving = system

    ismoving[1] &&
        (return extract_svector(move_coordinates!.acceleration, system, particle))

    return SVector(ntuple(_ -> 0.0, Val(ndims(system))))
end

@inline function particle_density(v, system::BoundarySPHSystem, particle)
    return particle_density(v, system.boundary_model, system, particle)
end

@inline function hydrodynamic_mass(system::BoundarySPHSystem, particle)
    return system.boundary_model.hydrodynamic_mass[particle]
end

@inline function smoothing_kernel(system::BoundarySPHSystem, distance)
    @unpack smoothing_kernel, smoothing_length = system.boundary_model
    return kernel(smoothing_kernel, distance, smoothing_length)
end

function update_positions!(system::BoundarySPHSystem, system_index, v, u, v_ode, u_ode,
                           semi, t)
    @unpack move_coordinates! = system

    move_coordinates!(system, t)
end

function update_quantities!(system::BoundarySPHSystem, system_index, v, u, v_ode, u_ode,
                            semi, t)
    @unpack boundary_model = system

    update_density!(boundary_model, system, system_index, v, u, v_ode, u_ode, semi)

    return system
end

# This update depends on the computed quantities of the fluid system and therefore
# has to be in `update_final!` after `update_quantities!`.
function update_final!(system::BoundarySPHSystem, system_index, v, u, v_ode, u_ode, semi, t)
    @unpack boundary_model = system

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
    @unpack density_calculator = system.boundary_model

    write_v0!(v0, density_calculator, system)
end

function write_v0!(v0, density_calculator, system::BoundarySPHSystem)
    return v0
end

function write_v0!(v0, ::ContinuityDensity, system::BoundarySPHSystem)
    @unpack cache = system.boundary_model
    @unpack initial_density = cache

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
    @unpack initial_density = model.cache

    for particle in eachparticle(system)
        initial_density[particle] = v[1, particle]
    end

    return system
end

"""
    BoundarySPHSystem(coordinates, model;
                      movement_function=nothing)

System for boundaries modeled by boundary particles.
The system is initialized with the coordinates of the particles and their masses.
The interaction between fluid and boundary particles is specified by the boundary model.

The `movement_function` is to define in which way the boundary particles move over time. Its
boolean return value is mandatory to determine in each timestep if the particles are moving or not.
This determines if the neighborhood search will be updated.
In the example below the `movement_function` only returns `true` (system is moving)
if the simulation time is lower than `0.1`.


# Examples
```julia
function movement_function(coordinates, t)

    if t < 0.1
        f(t) = 0.5*t^2 + t
        pos_1 = coordinates[2,1]
        pos_2 = f(t)
        diff_pos = pos_2 - pos_1
        coordinates[2,:] .+= diff_pos

        return true
    end

    return false
end
```
"""
struct BoundarySPHSystem{BM, NDIMS, ELTYPE <: Real, MF} <: System{NDIMS}
    coordinates       :: Array{ELTYPE, 2}
    boundary_model    :: BM
    movement_function :: MF
    ismoving          :: Vector{Bool}

    function BoundarySPHSystem(coordinates, model; movement_function=nothing)
        NDIMS = size(coordinates, 1)
        ismoving = zeros(Bool, 1)

        return new{typeof(model), NDIMS,
                   eltype(coordinates),
                   typeof(movement_function)}(coordinates, model,
                                              movement_function, ismoving)
    end
end

function Base.show(io::IO, system::BoundarySPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "BoundarySPHSystem{", ndims(system), "}(")
    print(io, system.boundary_model)
    print(io, ", ", system.movement_function)
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
        summary_line(io, "movement function", system.movement_function)
        summary_footer(io)
    end
end

@inline Base.eltype(system::BoundarySPHSystem) = eltype(system.coordinates)

# This does not account for moving boundaries, but it's only used to initialize the
# neighborhood search, anyway.
@inline initial_coordinates(system::BoundarySPHSystem) = system.coordinates

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
    # TODO moving boundaries
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

function update_quantities!(system::BoundarySPHSystem, system_index, v, u, v_ode, u_ode,
                            semi, t)
    @unpack boundary_model = system

    update_density!(boundary_model, system, system_index, v, u, v_ode, u_ode, semi)

    return system
end

# This update depends on the computed quantities of the fluid system and therefore
# has to be in `update_final!` after `update_quantities!`.
function update_final!(system::BoundarySPHSystem, system_index, v, u, v_ode, u_ode, semi, t)
    @unpack coordinates, movement_function, boundary_model = system

    system.ismoving[1] = move_boundary_particles!(movement_function, coordinates, t)

    update_pressure!(boundary_model, system, system_index, v, u, v_ode, u_ode, semi)

    return system
end

function move_boundary_particles!(movement_function, coordinates, t)
    movement_function(coordinates, t)
end

move_boundary_particles!(movement_function::Nothing, coordinates, t) = false

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

function system_viscosity(system::BoundarySPHSystem)
    return system.boundary_model.viscosity
end

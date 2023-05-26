"""
    BoundarySPHSystem(coordinates, mass, model;
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
    initial_coordinates :: Array{ELTYPE, 2}
    boundary_model      :: BM
    movement_function   :: MF
    ismoving            :: Vector{Bool}

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

# Note that we don't dispatch by `BoundarySPHSystem{BoundaryModel}` here because
# this is also used by the `TotalLagrangianSPHSystem`.
@inline function boundary_particle_impact(particle, boundary_particle,
                                          v_particle_system, v_boundary_system,
                                          particle_system, boundary_system,
                                          pos_diff, distance, m_b)
    @unpack boundary_model = boundary_system

    boundary_particle_impact(particle, boundary_particle,
                             boundary_model,
                             v_particle_system, v_boundary_system,
                             particle_system, boundary_system,
                             pos_diff, distance, m_b)
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

@inline function n_moving_particles(system::BoundarySPHSystem,
                                    density_calculator)
    return 0
end

@inline function n_moving_particles(system::BoundarySPHSystem,
                                    ::ContinuityDensity)
    nparticles(system)
end

@inline u_nvariables(system::BoundarySPHSystem) = 0

# For BoundaryModelDummyParticles with ContinuityDensity, this needs to be 1.
# For all other models and density calculators, it's irrelevant.
@inline v_nvariables(system::BoundarySPHSystem) = 1

@inline function current_coordinates(u, system::BoundarySPHSystem)
    return system.initial_coordinates
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

function update!(system::BoundarySPHSystem, system_index, v, u, v_ode, u_ode, semi, t)
    @unpack initial_coordinates, movement_function, boundary_model = system

    system.ismoving[1] = move_boundary_particles!(movement_function, initial_coordinates,
                                                  t)

    update!(boundary_model, system, system_index, v, u, v_ode, u_ode, semi)

    return system
end

function move_boundary_particles!(movement_function, coordinates, t)
    movement_function(coordinates, t)
end
move_boundary_particles!(movement_function::Nothing, coordinates, t) = false

@inline function update!(boundary_model::BoundaryModelMonaghanKajtar, system,
                         system_index, v, u, v_ode, u_ode, semi)
    # Nothing to do in the update step
    return boundary_model
end

@inline function update!(boundary_model::BoundaryModelDummyParticles, system, system_index,
                         v, u, v_ode, u_ode, semi)
    @unpack density_calculator = boundary_model

    compute_pressure1!(boundary_model, density_calculator, system, system_index, v, u,
                       v_ode, u_ode, semi)

    compute_density!(system, system_index, semi, u, u_ode, density_calculator)

    compute_pressure2!(boundary_model, density_calculator, system, v, semi)

    return boundary_model
end

function compute_pressure1!(boundary_model, density_calculator, system, system_index, v, u,
                            v_ode, u_ode, semi)
    return boundary_model
end

function compute_pressure1!(boundary_model, ::AdamiPressureExtrapolation, system,
                            system_index, v, u, v_ode, u_ode, semi)
    @unpack systems, neighborhood_searches = semi
    @unpack pressure, cache = boundary_model
    @unpack volume = cache

    pressure .= zero(eltype(pressure))
    volume .= zero(eltype(volume))

    # Use all other systems for the pressure extrapolation
    @trixi_timeit timer() "compute boundary pressure" foreach_enumerate(systems) do (neighbor_system_index,
                                                                                     neighbor_system)
        v_neighbor_system = wrap_v(v_ode, neighbor_system_index,
                                   neighbor_system, semi)
        u_neighbor_system = wrap_u(u_ode, neighbor_system_index,
                                   neighbor_system, semi)

        neighborhood_search = neighborhood_searches[system_index][neighbor_system_index]

        system_coords = current_coordinates(u, system)
        neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

        adami_pressure_extrapolation!(boundary_model, system, neighbor_system,
                                      system_coords, neighbor_coords,
                                      v_neighbor_system, neighborhood_search)
    end

    pressure ./= volume
end

function compute_pressure2!(boundary_model, density_calculator, system, v, semi)
    @unpack pressure, state_equation = boundary_model

    for particle in eachparticle(system)
        pressure[particle] = state_equation(particle_density(v, boundary_model, particle))
    end
end

function compute_pressure2!(boundary_model, ::AdamiPressureExtrapolation, system, v, semi)
    return boundary_model
end

@inline function adami_pressure_extrapolation!(boundary_model, system,
                                               neighbor_system::WeaklyCompressibleSPHSystem,
                                               system_coords, neighbor_coords,
                                               v_neighbor_system, neighborhood_search)
    @unpack pressure, cache = boundary_model
    @unpack volume = cache

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    for_particle_neighbor(system, neighbor_system,
                          system_coords, neighbor_coords,
                          neighborhood_search;
                          particles=eachparticle(system)) do particle, neighbor,
                                                             pos_diff, distance
        density_neighbor = particle_density(v_neighbor_system, neighbor_system,
                                            neighbor)

        # TODO moving boundaries
        pressure[particle] += (neighbor_system.pressure[neighbor] +
                               dot(neighbor_system.acceleration,
                                   density_neighbor * pos_diff)) *
                              smoothing_kernel(boundary_model, distance)
        volume[particle] += smoothing_kernel(boundary_model, distance)
    end

    # Limit pressure to be non-negative to avoid negative pressures at free surfaces
    for particle in eachparticle(system)
        pressure[particle] = max(pressure[particle], 0.0)
    end
end

@inline function adami_pressure_extrapolation!(boundary_model, system, neighbor_system,
                                               system_coords, neighbor_coords,
                                               v_neighbor_system, neighborhood_search)
    return boundary_model
end

@inline function compute_density!(system::BoundarySPHSystem, system_index, semi, u, u_ode,
                                  ::SummationDensity)
    @unpack boundary_model = system
    @unpack density = boundary_model.cache # Density is in the cache for SummationDensity

    summation_density!(system, system_index, semi, u, u_ode, density,
                       particles=eachparticle(system))
end

@inline function compute_density!(system::BoundarySPHSystem, system_index, semi, u, u_ode,
                                  ::AdamiPressureExtrapolation)
    @unpack boundary_model = system
    @unpack pressure, state_equation, cache = boundary_model
    @unpack density = cache

    density .= zero(eltype(density))

    for particle in eachparticle(system)
        density[particle] = inverse_state_equation(state_equation, pressure[particle])
    end
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

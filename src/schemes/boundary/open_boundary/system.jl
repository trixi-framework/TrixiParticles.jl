@doc raw"""
    OpenBoundarySPHSystem(boundary_zone::BoundaryZone;
                          fluid_system::FluidSystem, buffer_size::Integer,
                          boundary_model)

Open boundary system for in- and outflow particles.

# Arguments
- `boundary_zone`: See [`BoundaryZone`](@ref).

# Keywords
- `fluid_system`: The corresponding fluid system
- `boundary_model`: Boundary model (see [Open Boundary Models](@ref open_boundary_models))
- `buffer_size`: Number of buffer particles.

!!! warning "Experimental Implementation"
    This is an experimental feature and may change in any future releases.
"""
struct OpenBoundarySPHSystem{BM, ELTYPE, NDIMS, IC, FS, FSI, ARRAY1D, BC, FC, BZI, BZ,
                             B, C} <: AbstractSystem{NDIMS}
    boundary_model        :: BM
    initial_condition     :: IC
    fluid_system          :: FS
    fluid_system_index    :: FSI
    smoothing_length      :: ELTYPE
    mass                  :: ARRAY1D # Array{ELTYPE, 1}: [particle]
    density               :: ARRAY1D # Array{ELTYPE, 1}: [particle]
    volume                :: ARRAY1D # Array{ELTYPE, 1}: [particle]
    pressure              :: ARRAY1D # Array{ELTYPE, 1}: [particle]
    boundary_candidates   :: BC      # Array{Bool, 1}: [particle]
    fluid_candidates      :: FC      # Array{Bool, 1}: [particle]
    boundary_zone_indices :: BZI     # Array{UInt8, 1}: [particle]
    boundary_zones        :: BZ
    buffer                :: B
    cache                 :: C
end

function OpenBoundarySPHSystem(boundary_model, initial_condition, fluid_system,
                               fluid_system_index, smoothing_length, mass, density, volume,
                               pressure, boundary_candidates, fluid_candidates,
                               boundary_zone_indices, boundary_zone, buffer, cache)
    OpenBoundarySPHSystem{typeof(boundary_model), eltype(mass), ndims(initial_condition),
                          typeof(initial_condition), typeof(fluid_system),
                          typeof(fluid_system_index), typeof(mass),
                          typeof(boundary_candidates), typeof(fluid_candidates),
                          typeof(boundary_zone_indices), typeof(boundary_zone),
                          typeof(buffer),
                          typeof(cache)}(boundary_model, initial_condition, fluid_system,
                                         fluid_system_index, smoothing_length, mass,
                                         density, volume, pressure, boundary_candidates,
                                         fluid_candidates, boundary_zone_indices,
                                         boundary_zone, buffer, cache)
end

function OpenBoundarySPHSystem(boundary_zones::Union{BoundaryZone, Nothing}...;
                               fluid_system::AbstractFluidSystem, buffer_size::Integer,
                               boundary_model)
    boundary_zones_ = filter(bz -> !isnothing(bz), boundary_zones)
    reference_values_ = map(bz -> bz.reference_values, boundary_zones_)

    initial_conditions = union((bz.initial_condition for bz in boundary_zones)...)

    buffer = SystemBuffer(nparticles(initial_conditions), buffer_size)

    initial_conditions = allocate_buffer(initial_conditions, buffer)

    pressure = copy(initial_conditions.pressure)
    mass = copy(initial_conditions.mass)
    density = copy(initial_conditions.density)
    volume = similar(initial_conditions.density)

    cache = create_cache_open_boundary(boundary_model, initial_conditions,
                                       reference_values_)

    fluid_system_index = Ref(0)

    smoothing_length = initial_smoothing_length(fluid_system)

    boundary_candidates = fill(false, nparticles(initial_conditions))
    fluid_candidates = fill(false, nparticles(fluid_system))

    boundary_zone_indices = zeros(Int, nparticles(initial_conditions))

    # Create new `BoundaryZone`s with `reference_values` set to `nothing` for type stability.
    # `reference_values` are only used as API feature to temporarily store the reference values
    # in the `BoundaryZone`, but they are not used in the actual simulation.
    boundary_zones_new = map(zone -> BoundaryZone(zone.initial_condition,
                                                  zone.spanning_set,
                                                  zone.zone_origin,
                                                  zone.zone_width,
                                                  zone.flow_direction,
                                                  zone.plane_normal,
                                                  nothing,
                                                  zone.average_inflow_velocity,
                                                  zone.prescribed_density,
                                                  zone.prescribed_pressure,
                                                  zone.prescribed_velocity),
                             boundary_zones)

    return OpenBoundarySPHSystem(boundary_model, initial_conditions, fluid_system,
                                 fluid_system_index, smoothing_length, mass, density,
                                 volume, pressure, boundary_candidates, fluid_candidates,
                                 boundary_zone_indices, boundary_zones_new, buffer, cache)
end

function initialize!(system::OpenBoundarySPHSystem, semi)
    (; boundary_zones) = system

    update_boundary_zone_indices!(system, initial_coordinates(system), boundary_zones, semi)

    return system
end

function create_cache_open_boundary(boundary_model, initial_condition, reference_values)
    ELTYPE = eltype(initial_condition)

    # Separate `reference_values` into pressure, density and velocity reference values
    pressure_reference_values = map(ref -> ref.reference_pressure, reference_values)
    density_reference_values = map(ref -> ref.reference_density, reference_values)
    velocity_reference_values = map(ref -> ref.reference_velocity, reference_values)

    if boundary_model isa BoundaryModelCharacteristicsLastiwka
        characteristics = zeros(ELTYPE, 3, nparticles(initial_condition))
        previous_characteristics = zeros(ELTYPE, 3, nparticles(initial_condition))

        return (; characteristics=characteristics,
                previous_characteristics=previous_characteristics,
                pressure_reference_values=pressure_reference_values,
                density_reference_values=density_reference_values,
                velocity_reference_values=velocity_reference_values)
    else
        return (; pressure_reference_values=pressure_reference_values,
                density_reference_values=density_reference_values,
                velocity_reference_values=velocity_reference_values)
    end
end

timer_name(::OpenBoundarySPHSystem) = "open_boundary"
vtkname(system::OpenBoundarySPHSystem) = "open_boundary"

function Base.show(io::IO, system::OpenBoundarySPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "OpenBoundarySPHSystem{", ndims(system), "}(")
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::OpenBoundarySPHSystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "OpenBoundarySPHSystem{$(ndims(system))}")
        summary_line(io, "#particles", nparticles(system))
        summary_line(io, "#buffer_particles", system.buffer.buffer_size)
        summary_line(io, "#boundary_zones", length(system.boundary_zones))
        summary_line(io, "fluid system", type2string(system.fluid_system))
        summary_line(io, "boundary model", type2string(system.boundary_model))
        summary_footer(io)
    end
end

@inline function Base.eltype(::OpenBoundarySPHSystem{<:Any, ELTYPE}) where {ELTYPE}
    return ELTYPE
end

@inline buffer(system::OpenBoundarySPHSystem) = system.buffer

# The `UpdateCallback` is required to update particle positions between time steps
@inline requires_update_callback(system::OpenBoundarySPHSystem) = true

function smoothing_length(system::OpenBoundarySPHSystem, particle)
    return system.smoothing_length
end

@inline hydrodynamic_mass(system::OpenBoundarySPHSystem, particle) = system.mass[particle]

@inline function current_density(v, system::OpenBoundarySPHSystem)
    return system.density
end

@inline function current_pressure(v, system::OpenBoundarySPHSystem)
    return system.pressure
end

@inline function set_particle_pressure!(v, system::OpenBoundarySPHSystem, particle,
                                        pressure)
    system.pressure[particle] = pressure

    return v
end

@inline function set_particle_density!(v, system::OpenBoundarySPHSystem, particle,
                                       density)
    system.density[particle] = density

    return v
end

function update_boundary_interpolation!(system::OpenBoundarySPHSystem, v, u, v_ode, u_ode,
                                        semi, t)
    update_boundary_model!(system, system.boundary_model, v, u, v_ode, u_ode, semi, t)
end

# This function is called by the `UpdateCallback`, as the integrator array might be modified
function update_open_boundary_eachstep!(system::OpenBoundarySPHSystem, v_ode, u_ode,
                                        semi, t)
    (; boundary_model) = system

    u = wrap_u(u_ode, system, semi)
    v = wrap_v(v_ode, system, semi)

    @trixi_timeit timer() "check domain" check_domain!(system, v, u, v_ode, u_ode, semi)

    # Update density, pressure and velocity based on the specific boundary model
    @trixi_timeit timer() "update boundary quantities" begin
        update_boundary_quantities!(system, boundary_model, v, u, v_ode, u_ode, semi, t)
    end

    return system
end

update_open_boundary_eachstep!(system, v_ode, u_ode, semi, t) = system

function check_domain!(system, v, u, v_ode, u_ode, semi)
    (; boundary_zones, boundary_candidates, fluid_candidates, fluid_system) = system

    u_fluid = wrap_u(u_ode, fluid_system, semi)
    v_fluid = wrap_v(v_ode, fluid_system, semi)

    boundary_candidates .= false

    # Check the boundary particles whether they're leaving the boundary zone
    @threaded semi for particle in each_moving_particle(system)
        particle_coords = current_coords(u, system, particle)

        # Check if boundary particle is outside the boundary zone
        boundary_zone = current_boundary_zone(system, particle)
        if !is_in_boundary_zone(boundary_zone, particle_coords)
            boundary_candidates[particle] = true
        end
    end

    crossed_boundary_particles = findall(boundary_candidates)
    available_fluid_particles = findall(==(false), fluid_system.buffer.active_particle)

    @assert length(crossed_boundary_particles)<=length(available_fluid_particles) "Not enough fluid buffer particles available"

    # Convert open boundary particles in the fluid domain to fluid particles
    @threaded semi for i in eachindex(crossed_boundary_particles)
        particle = crossed_boundary_particles[i]
        particle_new = available_fluid_particles[i]

        boundary_zone = current_boundary_zone(system, particle)
        convert_particle!(system, fluid_system, boundary_zone, particle, particle_new,
                          v, u, v_fluid, u_fluid)
    end

    update_system_buffer!(system.buffer, semi)
    update_system_buffer!(fluid_system.buffer, semi)

    fluid_candidates .= false

    # Check the fluid particles whether they're entering the boundary zone
    @threaded semi for fluid_particle in each_moving_particle(fluid_system)
        fluid_coords = current_coords(u_fluid, fluid_system, fluid_particle)

        # Check if fluid particle is in any boundary zone
        for boundary_zone in boundary_zones
            if is_in_boundary_zone(boundary_zone, fluid_coords)
                fluid_candidates[fluid_particle] = true
            end
        end
    end

    crossed_fluid_particles = findall(fluid_candidates)
    available_boundary_particles = findall(==(false), system.buffer.active_particle)

    @assert length(crossed_fluid_particles)<=length(available_boundary_particles) "Not enough boundary buffer particles available"

    # Convert fluid particles in the open boundary zone to open boundary particles
    @threaded semi for i in eachindex(crossed_fluid_particles)
        particle = crossed_fluid_particles[i]
        particle_new = available_boundary_particles[i]

        convert_particle!(fluid_system, system, particle, particle_new,
                          v, u, v_fluid, u_fluid)
    end

    update_system_buffer!(system.buffer, semi)
    update_system_buffer!(fluid_system.buffer, semi)

    # Since particles have been transferred, the neighborhood searches must be updated
    update_nhs!(semi, u_ode)

    update_boundary_zone_indices!(system, u, boundary_zones, semi)

    return system
end

# Buffer particle is outside the boundary zone
@inline function convert_particle!(system::OpenBoundarySPHSystem, fluid_system,
                                   boundary_zone, particle, particle_new,
                                   v, u, v_fluid, u_fluid)
    relative_position = current_coords(u, system, particle) - boundary_zone.zone_origin

    # Check if particle is in- or outside the fluid domain.
    # `plane_normal` is always pointing into the fluid domain.
    if signbit(dot(relative_position, boundary_zone.plane_normal))
        deactivate_particle!(system, particle, u)

        return system
    end

    # Activate a new particle in simulation domain
    transfer_particle!(fluid_system, system, particle, particle_new, v_fluid, u_fluid, v, u)

    # Reset position of boundary particle
    for dim in 1:ndims(system)
        u[dim, particle] += boundary_zone.spanning_set[1][dim]
    end

    return system
end

# Fluid particle is in boundary zone
@inline function convert_particle!(fluid_system::AbstractFluidSystem, system,
                                   particle, particle_new, v, u, v_fluid, u_fluid)
    # Activate particle in boundary zone
    transfer_particle!(system, fluid_system, particle, particle_new, v, u, v_fluid, u_fluid)

    # Deactivate particle in interior domain
    deactivate_particle!(fluid_system, particle, u_fluid)

    return fluid_system
end

@inline function transfer_particle!(system_new, system_old, particle_old, particle_new,
                                    v_new, u_new, v_old, u_old)
    # Activate new particle
    system_new.buffer.active_particle[particle_new] = true

    # Transfer densities
    density = current_density(v_old, system_old, particle_old)
    set_particle_density!(v_new, system_new, particle_new, density)

    # Transfer pressure
    pressure = current_pressure(v_old, system_old, particle_old)
    set_particle_pressure!(v_new, system_new, particle_new, pressure)

    # Exchange position and velocity
    for dim in 1:ndims(system_new)
        u_new[dim, particle_new] = u_old[dim, particle_old]
        v_new[dim, particle_new] = v_old[dim, particle_old]
    end

    return system_new
end

function write_v0!(v0, system::OpenBoundarySPHSystem)
    # This is as fast as a loop with `@inbounds`, but it's GPU-compatible
    indices = CartesianIndices(system.initial_condition.velocity)
    copyto!(v0, indices, system.initial_condition.velocity, indices)

    return v0
end

function write_u0!(u0, system::OpenBoundarySPHSystem)
    (; initial_condition) = system

    # This is as fast as a loop with `@inbounds`, but it's GPU-compatible
    indices = CartesianIndices(initial_condition.coordinates)
    copyto!(u0, indices, initial_condition.coordinates, indices)

    return u0
end

# To account for boundary effects in the viscosity term of the RHS, use the viscosity model
# of the neighboring particle systems.
@inline function viscosity_model(system::OpenBoundarySPHSystem,
                                 neighbor_system::AbstractFluidSystem)
    return neighbor_system.viscosity
end

@inline function viscosity_model(system::OpenBoundarySPHSystem,
                                 neighbor_system::AbstractBoundarySystem)
    return neighbor_system.boundary_model.viscosity
end

# When the neighbor is an open boundary system, just use the viscosity of the fluid `system` instead
@inline viscosity_model(system, neighbor_system::OpenBoundarySPHSystem) = system.viscosity

function system_data(system::OpenBoundarySPHSystem, dv_ode, du_ode, v_ode, u_ode, semi)
    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    coordinates = current_coordinates(u, system)
    velocity = current_velocity(v, system)
    density = current_density(v, system)
    pressure = current_pressure(v, system)

    return (; coordinates, velocity, density, pressure)
end

function available_data(::OpenBoundarySPHSystem)
    return (:coordinates, :velocity, :density, :pressure)
end

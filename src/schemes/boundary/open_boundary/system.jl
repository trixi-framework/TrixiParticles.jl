@doc raw"""
    OpenBoundarySPHSystem(boundary_zone::BoundaryZone;
                          fluid_system::FluidSystem, buffer_size::Integer,
                          boundary_model,
                          reference_velocity=nothing,
                          reference_pressure=nothing,
                          reference_density=nothing)

Open boundary system for in- and outflow particles.

# Arguments
- `boundary_zone`: See [`BoundaryZone`](@ref).

# Keywords
- `fluid_system`: The corresponding fluid system
- `boundary_model`: Boundary model (see [Open Boundary Models](@ref open_boundary_models))
- `buffer_size`: Number of buffer particles.
- `reference_velocity`: Reference velocity is either a function mapping each particle's coordinates
                        and time to its velocity, an array where the ``i``-th column holds
                        the velocity of particle ``i`` or, for a constant fluid velocity,
                        a vector holding this velocity.
- `reference_pressure`: Reference pressure is either a function mapping each particle's coordinates
                        and time to its pressure, a vector holding the pressure of each particle,
                        or a scalar for a constant pressure over all particles.
- `reference_density`: Reference density is either a function mapping each particle's coordinates
                       and time to its density, a vector holding the density of each particle,
                       or a scalar for a constant density over all particles.

!!! note "Note"
    The reference values (`reference_velocity`, `reference_pressure`, `reference_density`)
    can also be set to `nothing`.
    In this case, they will either be extrapolated from the fluid domain ([BoundaryModelTafuni](@ref BoundaryModelTafuni))
    or evolved using the characteristic flow variables ([BoundaryModelLastiwka](@ref BoundaryModelLastiwka)).

!!! warning "Experimental Implementation"
    This is an experimental feature and may change in future releases.
    It is GPU-compatible (e.g., with CUDA.jl and AMDGPU.jl), but currently **not** supported with Metal.jl.
"""
struct OpenBoundarySPHSystem{BM, ELTYPE, NDIMS, IC, FS, FSI, ARRAY1D, BC, FC, BZ, RV,
                             RP, RD, B, UCU, C} <: System{NDIMS}
    boundary_model       :: BM
    initial_condition    :: IC
    fluid_system         :: FS
    fluid_system_index   :: FSI
    smoothing_length     :: ELTYPE
    mass                 :: ARRAY1D # Array{ELTYPE, 1}: [particle]
    density              :: ARRAY1D # Array{ELTYPE, 1}: [particle]
    volume               :: ARRAY1D # Array{ELTYPE, 1}: [particle]
    pressure             :: ARRAY1D # Array{ELTYPE, 1}: [particle]
    boundary_candidates  :: BC      # Array{UInt32, 1}: [particle]
    fluid_candidates     :: FC      # Array{UInt32, 1}: [particle]
    boundary_zone        :: BZ
    reference_velocity   :: RV
    reference_pressure   :: RP
    reference_density    :: RD
    buffer               :: B
    update_callback_used :: UCU
    cache                :: C
end

function OpenBoundarySPHSystem(boundary_model, initial_condition, fluid_system,
                               fluid_system_index, smoothing_length, mass, density, volume,
                               pressure, boundary_candidates, fluid_candidates,
                               boundary_zone, reference_velocity,
                               reference_pressure, reference_density, buffer,
                               update_callback_used, cache)
    OpenBoundarySPHSystem{typeof(boundary_model), eltype(mass), ndims(initial_condition),
                          typeof(initial_condition), typeof(fluid_system),
                          typeof(fluid_system_index), typeof(mass),
                          typeof(boundary_candidates), typeof(fluid_candidates),
                          typeof(boundary_zone), typeof(reference_velocity),
                          typeof(reference_pressure), typeof(reference_density),
                          typeof(buffer), typeof(update_callback_used),
                          typeof(cache)}(boundary_model, initial_condition, fluid_system,
                                         fluid_system_index, smoothing_length, mass,
                                         density, volume, pressure, boundary_candidates,
                                         fluid_candidates, boundary_zone,
                                         reference_velocity, reference_pressure,
                                         reference_density, buffer, update_callback_used,
                                         cache)
end

function OpenBoundarySPHSystem(boundary_zone::BoundaryZone;
                               fluid_system::FluidSystem,
                               buffer_size::Integer, boundary_model,
                               reference_velocity=nothing,
                               reference_pressure=nothing,
                               reference_density=nothing)
    (; initial_condition) = boundary_zone

    buffer = SystemBuffer(nparticles(initial_condition), buffer_size)

    initial_condition = allocate_buffer(initial_condition, buffer)

    NDIMS = ndims(initial_condition)

    pressure = copy(initial_condition.pressure)
    mass = copy(initial_condition.mass)
    density = copy(initial_condition.density)
    volume = similar(initial_condition.density)

    if !(reference_velocity isa Function || isnothing(reference_velocity) ||
         (reference_velocity isa Vector && length(reference_velocity) == NDIMS))
        throw(ArgumentError("`reference_velocity` must be either a function mapping " *
                            "each particle's coordinates and time to its velocity, " *
                            "an array where the ``i``-th column holds the velocity of particle ``i`` " *
                            "or, for a constant fluid velocity, a vector of length $NDIMS for a $(NDIMS)D problem holding this velocity"))
    else
        if reference_velocity isa Function
            test_result = reference_velocity(zeros(NDIMS), 0.0)
            if length(test_result) != NDIMS
                throw(ArgumentError("`reference_velocity` function must be of dimension $NDIMS"))
            end
        end
        reference_velocity_ = wrap_reference_function(reference_velocity, Val(NDIMS))
    end

    if !(reference_pressure isa Function || reference_pressure isa Real ||
         isnothing(reference_pressure))
        throw(ArgumentError("`reference_pressure` must be either a function mapping " *
                            "each particle's coordinates and time to its pressure, " *
                            "a vector holding the pressure of each particle, or a scalar"))
    else
        if reference_pressure isa Function
            test_result = reference_pressure(zeros(NDIMS), 0.0)
            if length(test_result) != 1
                throw(ArgumentError("`reference_pressure` function must be a scalar function"))
            end
        end
        reference_pressure_ = wrap_reference_function(reference_pressure, Val(NDIMS))
    end

    if !(reference_density isa Function || reference_density isa Real ||
         isnothing(reference_density))
        throw(ArgumentError("`reference_density` must be either a function mapping " *
                            "each particle's coordinates and time to its density, " *
                            "a vector holding the density of each particle, or a scalar"))
    else
        if reference_density isa Function
            test_result = reference_density(zeros(NDIMS), 0.0)
            if length(test_result) != 1
                throw(ArgumentError("`reference_density` function must be a scalar function"))
            end
        end
        reference_density_ = wrap_reference_function(reference_density, Val(NDIMS))
    end

    cache = create_cache_open_boundary(boundary_model, initial_condition,
                                       reference_density, reference_velocity,
                                       reference_pressure)

    # These will be set later
    update_callback_used = Ref(false)
    fluid_system_index = Ref(0)

    smoothing_length = initial_smoothing_length(fluid_system)

    boundary_candidates = fill(false, nparticles(initial_condition))
    fluid_candidates = fill(false, nparticles(fluid_system))

    return OpenBoundarySPHSystem(boundary_model, initial_condition, fluid_system,
                                 fluid_system_index, smoothing_length, mass, density,
                                 volume, pressure, boundary_candidates, fluid_candidates,
                                 boundary_zone, reference_velocity_,
                                 reference_pressure_, reference_density_, buffer,
                                 update_callback_used, cache)
end

function create_cache_open_boundary(boundary_model, initial_condition,
                                    reference_density, reference_velocity,
                                    reference_pressure)
    ELTYPE = eltype(initial_condition)

    prescribed_pressure = isnothing(reference_pressure) ? false : true
    prescribed_velocity = isnothing(reference_velocity) ? false : true
    prescribed_density = isnothing(reference_density) ? false : true

    if boundary_model isa BoundaryModelTafuni
        return (; prescribed_pressure=prescribed_pressure,
                prescribed_density=prescribed_density,
                prescribed_velocity=prescribed_velocity)
    end

    characteristics = zeros(ELTYPE, 3, nparticles(initial_condition))
    previous_characteristics = zeros(ELTYPE, 3, nparticles(initial_condition))

    return (; characteristics=characteristics,
            previous_characteristics=previous_characteristics,
            prescribed_pressure=prescribed_pressure,
            prescribed_density=prescribed_density, prescribed_velocity=prescribed_velocity)
end

timer_name(::OpenBoundarySPHSystem) = "open_boundary"
vtkname(system::OpenBoundarySPHSystem) = "open_boundary"
boundary_type_name(::BoundaryZone{ZT}) where {ZT} = string(nameof(ZT))

function Base.show(io::IO, system::OpenBoundarySPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "OpenBoundarySPHSystem{", ndims(system), "}(")
    print(io, boundary_type_name(system.boundary_zone))
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
        summary_line(io, "fluid system", type2string(system.fluid_system))
        summary_line(io, "boundary model", type2string(system.boundary_model))
        summary_line(io, "boundary type", boundary_type_name(system.boundary_zone))
        summary_line(io, "prescribed velocity", type2string(system.reference_velocity))
        summary_line(io, "prescribed pressure", type2string(system.reference_pressure))
        summary_line(io, "prescribed density", type2string(system.reference_density))
        summary_line(io, "width", round(system.boundary_zone.zone_width, digits=3))
        summary_footer(io)
    end
end

@inline function Base.eltype(::OpenBoundarySPHSystem{<:Any, ELTYPE}) where {ELTYPE}
    return ELTYPE
end

function reset_callback_flag!(system::OpenBoundarySPHSystem)
    system.update_callback_used[] = false

    return system
end

update_callback_used!(system::OpenBoundarySPHSystem) = system.update_callback_used[] = true

function corresponding_fluid_system(system::OpenBoundarySPHSystem, semi)
    return system.fluid_system
end

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

function update_boundary_interpolation!(system::OpenBoundarySPHSystem, v, u, v_ode, u_ode,
                                        semi, t; update_from_callback=false)
    if !update_from_callback && !(system.update_callback_used[])
        throw(ArgumentError("`UpdateCallback` is required when using `OpenBoundarySPHSystem`"))
    end

    update_boundary_model!(system, system.boundary_model, v, u, v_ode, u_ode, semi, t)
end

# This function is called by the `UpdateCallback`, as the integrator array might be modified
function update_open_boundary_eachstep!(system::OpenBoundarySPHSystem, v_ode, u_ode,
                                        semi, t)
    u = wrap_u(u_ode, system, semi)
    v = wrap_v(v_ode, system, semi)

    @trixi_timeit timer() "check domain" check_domain!(system, v, u, v_ode, u_ode, semi)

    # Update density, pressure and velocity based on the characteristic variables.
    # See eq. 13-15 in Lastiwka (2009) https://doi.org/10.1002/fld.1971
    @trixi_timeit timer() "update boundary quantities" update_boundary_quantities!(system,
                                                                                   system.boundary_model,
                                                                                   v, u,
                                                                                   v_ode,
                                                                                   u_ode,
                                                                                   semi, t)

    return system
end

update_open_boundary_eachstep!(system, v_ode, u_ode, semi, t) = system

function check_domain!(system, v, u, v_ode, u_ode, semi)
    (; boundary_zone, boundary_candidates, fluid_candidates) = system
    fluid_system = corresponding_fluid_system(system, semi)

    u_fluid = wrap_u(u_ode, fluid_system, semi)
    v_fluid = wrap_v(v_ode, fluid_system, semi)

    boundary_candidates .= false

    # Check the boundary particles whether they're leaving the boundary zone
    @threaded semi for particle in each_moving_particle(system)
        particle_coords = current_coords(u, system, particle)

        # Check if boundary particle is outside the boundary zone
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

        convert_particle!(system, fluid_system, boundary_zone, particle, particle_new,
                          v, u, v_fluid, u_fluid)
    end

    update_system_buffer!(system.buffer, semi)
    update_system_buffer!(fluid_system.buffer, semi)

    fluid_candidates .= false

    # Check the fluid particles whether they're entering the boundary zone
    @threaded semi for fluid_particle in each_moving_particle(fluid_system)
        fluid_coords = current_coords(u_fluid, fluid_system, fluid_particle)

        # Check if fluid particle is in boundary zone
        if is_in_boundary_zone(boundary_zone, fluid_coords)
            fluid_candidates[fluid_particle] = true
        end
    end

    crossed_fluid_particles = findall(fluid_candidates)
    available_boundary_particles = findall(==(false), system.buffer.active_particle)

    @assert length(crossed_fluid_particles)<=length(available_boundary_particles) "Not enough boundary buffer particles available"

    # Convert fluid particles in the open boundary zone to open boundary particles
    @threaded semi for i in eachindex(crossed_fluid_particles)
        particle = crossed_fluid_particles[i]
        particle_new = available_boundary_particles[i]

        convert_particle!(fluid_system, system, boundary_zone, particle, particle_new,
                          v, u, v_fluid, u_fluid)
    end

    update_system_buffer!(system.buffer, semi)
    update_system_buffer!(fluid_system.buffer, semi)

    # Since particles have been transferred, the neighborhood searches must be updated
    update_nhs!(semi, u_ode)

    return system
end

# Outflow particle is outside the boundary zone
@inline function convert_particle!(system::OpenBoundarySPHSystem, fluid_system,
                                   boundary_zone::BoundaryZone{OutFlow}, particle,
                                   particle_new, v, u, v_fluid, u_fluid)
    deactivate_particle!(system, particle, u)

    return system
end

# Inflow particle is outside the boundary zone
@inline function convert_particle!(system::OpenBoundarySPHSystem, fluid_system,
                                   boundary_zone::BoundaryZone{InFlow}, particle,
                                   particle_new, v, u, v_fluid, u_fluid)
    (; spanning_set) = boundary_zone

    # Activate a new particle in simulation domain
    transfer_particle!(fluid_system, system, particle, particle_new, v_fluid, u_fluid, v, u)

    # Reset position of boundary particle
    for dim in 1:ndims(system)
        u[dim, particle] += spanning_set[1][dim]
    end

    return system
end

# Buffer particle is outside the boundary zone
@inline function convert_particle!(system::OpenBoundarySPHSystem, fluid_system,
                                   boundary_zone::BoundaryZone{BidirectionalFlow},
                                   particle, particle_new, v, u, v_fluid, u_fluid)
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
@inline function convert_particle!(fluid_system::FluidSystem, system,
                                   boundary_zone, particle, particle_new,
                                   v, u, v_fluid, u_fluid)
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

wrap_reference_function(::Nothing, ::Val) = nothing

function wrap_reference_function(function_::Function, ::Val)
    # Already a function
    return function_
end

# Name the function so that the summary box does know which kind of function this is
function wrap_reference_function(constant_scalar_::Number, ::Val)
    return constant_scalar(coords, t) = constant_scalar_
end

# For vectors and tuples
# Name the function so that the summary box does know which kind of function this is
function wrap_reference_function(constant_vector_, ::Val{NDIMS}) where {NDIMS}
    return constant_vector(coords, t) = SVector{NDIMS}(constant_vector_)
end

function reference_value(value::Function, quantity, position, t)
    return value(position, t)
end

# This method is used when extrapolating quantities from the domain
# instead of using the method of characteristics
reference_value(value::Nothing, quantity, position, t) = quantity

# To account for boundary effects in the viscosity term of the RHS, use the viscosity model
# of the neighboring particle systems.
@inline function viscosity_model(system::OpenBoundarySPHSystem,
                                 neighbor_system::FluidSystem)
    return neighbor_system.viscosity
end

@inline function viscosity_model(system::OpenBoundarySPHSystem,
                                 neighbor_system::BoundarySystem)
    return neighbor_system.boundary_model.viscosity
end

# When the neighbor is an open boundary system, just use the viscosity of the fluid `system` instead
@inline viscosity_model(system, neighbor_system::OpenBoundarySPHSystem) = system.viscosity

function system_data(system::OpenBoundarySPHSystem, v_ode, u_ode, semi)
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

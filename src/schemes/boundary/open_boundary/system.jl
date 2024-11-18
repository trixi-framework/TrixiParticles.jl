@doc raw"""
    OpenBoundarySPHSystem(boundary_zone::Union{InFlow, OutFlow};
                          fluid_system::FluidSystem, buffer_size::Integer,
                          boundary_model,
                          reference_velocity=nothing,
                          reference_pressure=nothing,
                          reference_density=nothing)

Open boundary system for in- and outflow particles.

# Arguments
- `boundary_zone`: Use [`InFlow`](@ref) for an inflow and [`OutFlow`](@ref) for an outflow boundary.

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

!!! warning "Experimental Implementation"
	This is an experimental feature and may change in any future releases.
"""
struct OpenBoundarySPHSystem{BM, BZ, NDIMS, ELTYPE <: Real, IC, FS, ARRAY1D, RV,
                             RP, RD, B, C} <: System{NDIMS, IC}
    initial_condition    :: IC
    fluid_system         :: FS
    boundary_model       :: BM
    mass                 :: ARRAY1D # Array{ELTYPE, 1}: [particle]
    density              :: ARRAY1D # Array{ELTYPE, 1}: [particle]
    volume               :: ARRAY1D # Array{ELTYPE, 1}: [particle]
    pressure             :: ARRAY1D # Array{ELTYPE, 1}: [particle]
    boundary_zone        :: BZ
    flow_direction       :: SVector{NDIMS, ELTYPE}
    reference_velocity   :: RV
    reference_pressure   :: RP
    reference_density    :: RD
    buffer               :: B
    update_callback_used :: Ref{Bool}
    cache                :: C

    function OpenBoundarySPHSystem(boundary_zone::Union{InFlow, OutFlow};
                                   fluid_system::FluidSystem,
                                   buffer_size::Integer, boundary_model,
                                   reference_velocity=nothing,
                                   reference_pressure=nothing,
                                   reference_density=nothing)
        (; initial_condition) = boundary_zone

        check_reference_values!(boundary_model, reference_density, reference_pressure,
                                reference_velocity)

        buffer = SystemBuffer(nparticles(initial_condition), buffer_size)

        initial_condition = allocate_buffer(initial_condition, buffer)

        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)

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

        flow_direction_ = boundary_zone.flow_direction

        cache = create_cache_open_boundary(boundary_model, initial_condition)

        return new{typeof(boundary_model), typeof(boundary_zone), NDIMS, ELTYPE,
                   typeof(initial_condition), typeof(fluid_system), typeof(mass),
                   typeof(reference_velocity_), typeof(reference_pressure_),
                   typeof(reference_density_), typeof(buffer),
                   typeof(cache)}(initial_condition, fluid_system, boundary_model, mass,
                                  density, volume, pressure, boundary_zone,
                                  flow_direction_, reference_velocity_, reference_pressure_,
                                  reference_density_, buffer, false, cache)
    end
end

function create_cache_open_boundary(boundary_model, initial_condition)
    ELTYPE = eltype(initial_condition)

    characteristics = zeros(ELTYPE, 3, nparticles(initial_condition))
    previous_characteristics = zeros(ELTYPE, 3, nparticles(initial_condition))

    return (; characteristics=characteristics,
            previous_characteristics=previous_characteristics)
end

timer_name(::OpenBoundarySPHSystem) = "open_boundary"
vtkname(system::OpenBoundarySPHSystem) = "open_boundary"

function Base.show(io::IO, system::OpenBoundarySPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "OpenBoundarySPHSystem{", ndims(system), "}(")
    print(io, type2string(system.boundary_zone))
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
        summary_line(io, "boundary", type2string(system.boundary_zone))
        summary_line(io, "flow direction", system.flow_direction)
        summary_line(io, "prescribed velocity", type2string(system.reference_velocity))
        summary_line(io, "prescribed pressure", type2string(system.reference_pressure))
        summary_line(io, "prescribed density", type2string(system.reference_density))
        summary_line(io, "width", round(system.boundary_zone.zone_width, digits=3))
        summary_footer(io)
    end
end

function reset_callback_flag!(system::OpenBoundarySPHSystem)
    system.update_callback_used[] = false

    return system
end

update_callback_used!(system::OpenBoundarySPHSystem) = system.update_callback_used[] = true

@inline hydrodynamic_mass(system::OpenBoundarySPHSystem, particle) = system.mass[particle]

@inline function particle_density(v, system::OpenBoundarySPHSystem, particle)
    return system.density[particle]
end

@inline function particle_pressure(v, system::OpenBoundarySPHSystem, particle)
    return system.pressure[particle]
end

function update_final!(system::OpenBoundarySPHSystem, v, u, v_ode, u_ode, semi, t;
                       update_from_callback=false)
    if !update_from_callback && !(system.update_callback_used[])
        throw(ArgumentError("`UpdateCallback` is required when using `OpenBoundarySPHSystem`"))
    end

    update_final!(system, system.boundary_model, v, u, v_ode, u_ode, semi, t)
end

# This function is called by the `UpdateCallback`, as the integrator array might be modified
function update_open_boundary_eachstep!(system::OpenBoundarySPHSystem, v_ode, u_ode,
                                        semi, t)
    u = wrap_u(u_ode, system, semi)
    v = wrap_v(v_ode, system, semi)

    # Update density, pressure and velocity based on the characteristic variables.
    # See eq. 13-15 in Lastiwka (2009) https://doi.org/10.1002/fld.1971
    @trixi_timeit timer() "update boundary quantities" update_boundary_quantities!(system,
                                                                                   system.boundary_model,
                                                                                   v, u,
                                                                                   v_ode,
                                                                                   u_ode,
                                                                                   semi, t)

    @trixi_timeit timer() "check domain" check_domain!(system, v, u, v_ode, u_ode, semi)

    # Update buffers
    update_system_buffer!(system.buffer)
    update_system_buffer!(system.fluid_system.buffer)
end

update_open_boundary_eachstep!(system, v_ode, u_ode, semi, t) = system

function check_domain!(system, v, u, v_ode, u_ode, semi)
    (; boundary_zone, fluid_system) = system

    u_fluid = wrap_u(u_ode, fluid_system, semi)
    v_fluid = wrap_v(v_ode, fluid_system, semi)

    neighborhood_search = get_neighborhood_search(system, fluid_system, semi)

    for particle in each_moving_particle(system)
        particle_coords = current_coords(u, system, particle)

        # Check if boundary particle is outside the boundary zone
        if !is_in_boundary_zone(boundary_zone, particle_coords)
            convert_particle!(system, fluid_system, boundary_zone, particle,
                              v, u, v_fluid, u_fluid)
        end

        # Check the neighboring fluid particles whether they're entering the boundary zone
        for neighbor in PointNeighbors.eachneighbor(particle_coords, neighborhood_search)
            fluid_coords = current_coords(u_fluid, fluid_system, neighbor)

            # Check if neighboring fluid particle is in boundary zone
            if is_in_boundary_zone(boundary_zone, fluid_coords)
                convert_particle!(fluid_system, system, boundary_zone, neighbor,
                                  v, u, v_fluid, u_fluid)
            end
        end
    end

    return system
end

# Outflow particle is outside the boundary zone
@inline function convert_particle!(system::OpenBoundarySPHSystem, fluid_system,
                                   boundary_zone::OutFlow, particle, v, u,
                                   v_fluid, u_fluid)
    deactivate_particle!(system, particle, u)

    return system
end

# Inflow particle is outside the boundary zone
@inline function convert_particle!(system::OpenBoundarySPHSystem, fluid_system,
                                   boundary_zone::InFlow, particle, v, u,
                                   v_fluid, u_fluid)
    (; spanning_set) = boundary_zone

    # Activate a new particle in simulation domain
    transfer_particle!(fluid_system, system, particle, v_fluid, u_fluid, v, u)

    # Reset position of boundary particle
    for dim in 1:ndims(system)
        u[dim, particle] += spanning_set[1][dim]
    end

    return system
end

# Fluid particle is in boundary zone
@inline function convert_particle!(fluid_system::FluidSystem, system,
                                   boundary_zone, particle, v, u, v_fluid, u_fluid)
    # Activate particle in boundary zone
    transfer_particle!(system, fluid_system, particle, v, u, v_fluid, u_fluid)

    # Deactivate particle in interior domain
    deactivate_particle!(fluid_system, particle, u_fluid)

    return fluid_system
end

@inline function transfer_particle!(system_new, system_old, particle_old,
                                    v_new, u_new, v_old, u_old)
    particle_new = activate_next_particle(system_new)

    # Transfer densities
    density = particle_density(v_old, system_old, particle_old)
    set_particle_density!(v_new, system_new, particle_new, density)

    # Transfer pressure
    pressure = particle_pressure(v_old, system_old, particle_old)
    set_particle_pressure!(v_new, system_new, particle_new, pressure)

    # Exchange position and velocity
    for dim in 1:ndims(system_new)
        u_new[dim, particle_new] = u_old[dim, particle_old]
        v_new[dim, particle_new] = v_old[dim, particle_old]
    end

    return system_new
end

function write_v0!(v0, system::OpenBoundarySPHSystem)
    (; initial_condition) = system

    for particle in eachparticle(system)
        # Write particle velocities
        for dim in 1:ndims(system)
            v0[dim, particle] = initial_condition.velocity[dim, particle]
        end
    end

    return v0
end

function write_u0!(u0, system::OpenBoundarySPHSystem)
    (; initial_condition) = system

    for particle in eachparticle(system)
        # Write particle velocities
        for dim in 1:ndims(system)
            u0[dim, particle] = initial_condition.coordinates[dim, particle]
        end
    end

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

function reference_value(value::Function, quantity, system, particle, position, t)
    return value(position, t)
end

# This method is used when extrapolating quantities from the domain
# instead of using the method of characteristics
reference_value(value::Nothing, quantity, system, particle, position, t) = quantity

function check_reference_values!(boundary_model::BoundaryModelLastiwka,
                                 reference_density, reference_pressure, reference_velocity)
    # TODO: Extrapolate the reference values from the domain
    if any(isnothing.([reference_density, reference_pressure, reference_velocity]))
        throw(ArgumentError("for `BoundaryModelLastiwka` all reference values must be specified"))
    end

    return boundary_model
end

# To account for boundary effects in the viscosity term of the RHS, use the viscosity model
# of the neighboring particle systems.
@inline viscosity_model(system::OpenBoundarySPHSystem, neighbor_system::FluidSystem) = neighbor_system.viscosity
@inline viscosity_model(system::OpenBoundarySPHSystem, neighbor_system::BoundarySystem) = neighbor_system.boundary_model.viscosity
# When the neighbor is an open boundary system, just use the viscosity of the fluid `system` instead
@inline viscosity_model(system, neighbor_system::OpenBoundarySPHSystem) = system.viscosity

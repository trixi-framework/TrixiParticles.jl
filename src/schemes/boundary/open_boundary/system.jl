@doc raw"""
    OpenBoundarySystem(boundary_zone::BoundaryZone;
                       fluid_system::AbstractFluidSystem, buffer_size::Integer,
                       boundary_model)

Open boundary system for in- and outflow particles.

# Arguments
- `boundary_zone`: See [`BoundaryZone`](@ref).

# Keywords
- `fluid_system`: The corresponding fluid system
- `boundary_model`: Boundary model (see [Open Boundary Models](@ref open_boundary_models))
- `buffer_size`: Number of buffer particles.
- `pressure_acceleration`: Pressure acceleration formulation for the system. Required only
                           when using [`BoundaryModelDynamicalPressureZhang`](@ref).
                           Defaults to the formulation from `fluid_system` if applicable; otherwise, `nothing`.
- `shifting_technique`: [Shifting technique](@ref shifting) or [transport velocity formulation](@ref transport_velocity_formulation)
                        for this system. Defaults to the technique used by `fluid_system`.
                        As of now, only supported for [`BoundaryModelDynamicalPressureZhang`](@ref).

!!! warning "Experimental Implementation"
    This is an experimental feature and may change in any future releases.
"""
struct OpenBoundarySystem{BM, ELTYPE, NDIMS, IC, FS, FSI, K, ARRAY1D, BC, FC, BZI, BZ,
                          B, PF, ST, C} <: AbstractSystem{NDIMS}
    boundary_model                    :: BM
    initial_condition                 :: IC
    fluid_system                      :: FS
    fluid_system_index                :: FSI
    smoothing_kernel                  :: K
    smoothing_length                  :: ELTYPE
    mass                              :: ARRAY1D # Array{ELTYPE, 1}: [particle]
    volume                            :: ARRAY1D # Array{ELTYPE, 1}: [particle]
    boundary_candidates               :: BC      # Array{Bool, 1}: [particle]
    fluid_candidates                  :: FC      # Array{Bool, 1}: [particle]
    boundary_zone_indices             :: BZI     # Array{UInt8, 1}: [particle]
    boundary_zones                    :: BZ
    buffer                            :: B
    pressure_acceleration_formulation :: PF
    shifting_technique                :: ST
    cache                             :: C
end

function OpenBoundarySystem(boundary_model, initial_condition, fluid_system,
                            fluid_system_index, smoothing_kernel, smoothing_length, mass,
                            volume, boundary_candidates, fluid_candidates,
                            boundary_zone_indices, boundary_zone, buffer,
                            pressure_acceleration, shifting_technique, cache)
    OpenBoundarySystem{typeof(boundary_model), eltype(mass), ndims(initial_condition),
                       typeof(initial_condition), typeof(fluid_system),
                       typeof(fluid_system_index), typeof(smoothing_kernel), typeof(mass),
                       typeof(boundary_candidates), typeof(fluid_candidates),
                       typeof(boundary_zone_indices), typeof(boundary_zone), typeof(buffer),
                       typeof(pressure_acceleration), typeof(shifting_technique),
                       typeof(cache)}(boundary_model, initial_condition, fluid_system,
                                      fluid_system_index, smoothing_kernel,
                                      smoothing_length, mass, volume, boundary_candidates,
                                      fluid_candidates, boundary_zone_indices,
                                      boundary_zone, buffer, pressure_acceleration,
                                      shifting_technique, cache)
end

function OpenBoundarySystem(boundary_zones::Union{BoundaryZone, Nothing}...;
                            fluid_system::AbstractFluidSystem, buffer_size::Integer,
                            boundary_model,
                            pressure_acceleration=boundary_model isa
                                                  BoundaryModelDynamicalPressureZhang ?
                                                  fluid_system.pressure_acceleration_formulation :
                                                  nothing,
                            shifting_technique=boundary_model isa
                                               BoundaryModelDynamicalPressureZhang ?
                                               shifting_technique(fluid_system) : nothing)
    boundary_zones_ = filter(bz -> !isnothing(bz), boundary_zones)

    initial_conditions = union((bz.initial_condition for bz in boundary_zones_)...)

    buffer = SystemBuffer(nparticles(initial_conditions), buffer_size)

    initial_conditions = allocate_buffer(initial_conditions, buffer)

    mass = copy(initial_conditions.mass)
    volume = similar(initial_conditions.density)

    cache = (;
             create_cache_shifting(initial_conditions, shifting_technique)...,
             create_cache_open_boundary(boundary_model, fluid_system, initial_conditions,
                                        boundary_zones_)...)

    fluid_system_index = Ref(0)

    smoothing_kernel = system_smoothing_kernel(fluid_system)
    smoothing_length = initial_smoothing_length(fluid_system)

    boundary_candidates = fill(false, nparticles(initial_conditions))
    fluid_candidates = fill(false, nparticles(fluid_system))

    boundary_zone_indices = zeros(Int, nparticles(initial_conditions))

    # Create new `BoundaryZone`s with `reference_values` set to `nothing` for type stability.
    # `reference_values` are only used as API feature to temporarily store the reference values
    # in the `BoundaryZone`, but they are not used in the actual simulation.
    # The reference values are extracted above in the "create cache" function
    # and then stored in `system.cache` as a `Tuple`.
    boundary_zones_new = map(zone -> BoundaryZone(zone.initial_condition,
                                                  zone.spanning_set,
                                                  zone.zone_origin,
                                                  zone.zone_width,
                                                  zone.flow_direction,
                                                  zone.face_normal,
                                                  zone.rest_pressure,
                                                  nothing,
                                                  zone.average_inflow_velocity,
                                                  zone.prescribed_density,
                                                  zone.prescribed_pressure,
                                                  zone.prescribed_velocity),
                             boundary_zones_)

    return OpenBoundarySystem(boundary_model, initial_conditions, fluid_system,
                              fluid_system_index, smoothing_kernel, smoothing_length, mass,
                              volume, boundary_candidates, fluid_candidates,
                              boundary_zone_indices, boundary_zones_new, buffer,
                              pressure_acceleration, shifting_technique, cache)
end

function initialize!(system::OpenBoundarySystem, semi)
    (; boundary_zones) = system

    update_boundary_zone_indices!(system, initial_coordinates(system), boundary_zones, semi)

    return system
end

function create_cache_open_boundary(boundary_model, fluid_system, initial_condition,
                                    boundary_zones)
    reference_values = map(bz -> bz.reference_values, boundary_zones)
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
                pressure=copy(initial_condition.pressure),
                density=copy(initial_condition.density),
                pressure_reference_values=pressure_reference_values,
                density_reference_values=density_reference_values,
                velocity_reference_values=velocity_reference_values)
    elseif boundary_model isa BoundaryModelDynamicalPressureZhang
        # A separate array for the boundary pressure is required,
        # since it is specified independently from the computed pressure for the momentum equation.
        pressure_boundary = copy(initial_condition.pressure)

        # The first entry of the density vector can be used,
        # as it was already verified in `allocate_buffer` that the density array is constant.
        density_rest = first(initial_condition.density)

        dd = density_diffusion(fluid_system)
        if dd isa DensityDiffusionAntuono
            density_diffusion_ = DensityDiffusionAntuono(initial_condition; delta=dd.delta)
        else
            density_diffusion_ = dd
        end

        cache = (; density_calculator=ContinuityDensity(),
                 density_diffusion=density_diffusion_,
                 pressure_boundary=pressure_boundary,
                 density_rest=density_rest,
                 pressure_reference_values=pressure_reference_values,
                 density_reference_values=density_reference_values,
                 velocity_reference_values=velocity_reference_values)

        if fluid_system isa EntropicallyDampedSPHSystem
            # Density and pressure is stored in `v`
            return cache
        else
            # Only density is stored in `v`
            return (; pressure=copy(initial_condition.pressure), cache...)
        end
    else
        return (;
                pressure=copy(initial_condition.pressure),
                density=copy(initial_condition.density),
                pressure_reference_values=pressure_reference_values,
                density_reference_values=density_reference_values,
                velocity_reference_values=velocity_reference_values)
    end
end

timer_name(::OpenBoundarySystem) = "open_boundary"
vtkname(system::OpenBoundarySystem) = "open_boundary"

function Base.show(io::IO, system::OpenBoundarySystem)
    @nospecialize system # reduce precompilation time

    print(io, "OpenBoundarySystem{", ndims(system), "}(")
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::OpenBoundarySystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "OpenBoundarySystem{$(ndims(system))}")
        summary_line(io, "#particles", nparticles(system))
        summary_line(io, "#buffer_particles", system.buffer.buffer_size)
        summary_line(io, "#boundary_zones", length(system.boundary_zones))
        summary_line(io, "fluid system", type2string(system.fluid_system))
        summary_line(io, "boundary model", type2string(system.boundary_model))
        if system.boundary_model isa BoundaryModelDynamicalPressureZhang
            summary_line(io, "density diffusion", density_diffusion(system))
            summary_line(io, "shifting technique", shifting_technique(system))
        end
        for (i, pm) in enumerate(system.cache.pressure_reference_values)
            !isa(pm, AbstractPressureModel) && continue
            summary_line(io, "pressure model", type2string(pm) * " (in boundary zone $i)")
        end
        summary_footer(io)
    end
end

@inline function Base.eltype(::OpenBoundarySystem{<:Any, ELTYPE}) where {ELTYPE}
    return ELTYPE
end

@inline buffer(system::OpenBoundarySystem) = system.buffer

# The `UpdateCallback` is required to update particle positions between time steps
@inline requires_update_callback(system::OpenBoundarySystem) = true

function smoothing_length(system::OpenBoundarySystem, particle)
    return system.smoothing_length
end

@inline acceleration_source(system::OpenBoundarySystem) = system.fluid_system.acceleration

@inline function v_nvariables(system::OpenBoundarySystem)
    return ndims(system)
end

@inline function shifting_technique(system::OpenBoundarySystem)
    return system.shifting_technique
end

system_sound_speed(system::OpenBoundarySystem) = system_sound_speed(system.fluid_system)

@inline hydrodynamic_mass(system::OpenBoundarySystem, particle) = system.mass[particle]

@inline function current_density(v, system::OpenBoundarySystem)
    return system.cache.density
end

@inline function current_pressure(v, system::OpenBoundarySystem)
    return system.cache.pressure
end

@inline function set_particle_pressure!(v, system::OpenBoundarySystem, particle, pressure)
    current_pressure(v, system)[particle] = pressure

    return v
end

@inline function set_particle_density!(v, system::OpenBoundarySystem, particle, density)
    current_density(v, system)[particle] = density

    return v
end

@inline function add_velocity!(du, v, u, particle, system::OpenBoundarySystem, t)
    boundary_zone = current_boundary_zone(system, particle)

    pos = current_coords(u, system, particle)
    v_particle = reference_velocity(boundary_zone, v, system, particle, pos, t)

    # This is zero unless a shifting technique is used
    delta_v_ = delta_v(system, particle)

    for i in 1:ndims(system)
        @inbounds du[i, particle] = v_particle[i] + delta_v_[i]
    end

    return du
end

function update_boundary_interpolation!(system::OpenBoundarySystem, v, u, v_ode, u_ode,
                                        semi, t)
    update_boundary_model!(system, system.boundary_model, v, u, v_ode, u_ode, semi, t)
    update_shifting!(system, shifting_technique(system), v, u, v_ode, u_ode, semi)
end

# This function is called by the `UpdateCallback`, as the integrator array might be modified
function update_open_boundary_eachstep!(system::OpenBoundarySystem, v_ode, u_ode,
                                        semi, t, integrator)
    (; boundary_model) = system

    @trixi_timeit timer() "update open boundary" begin
        u = wrap_u(u_ode, system, semi)
        v = wrap_v(v_ode, system, semi)

        @trixi_timeit timer() "check domain" check_domain!(system, v, u, v_ode, u_ode, semi)

        update_pressure_model!(system, v, u, semi, integrator.dt)

        # Update density, pressure and velocity based on the specific boundary model
        @trixi_timeit timer() "update boundary quantities" begin
            update_boundary_quantities!(system, boundary_model, v, u, v_ode, u_ode, semi, t)
        end
    end

    # Tell OrdinaryDiffEq that `integrator.u` has been modified
    u_modified!(integrator, true)

    return system
end

update_open_boundary_eachstep!(system, v_ode, u_ode, semi, t, integrator) = system

function check_domain!(system, v, u, v_ode, u_ode, semi)
    (; boundary_zones, boundary_candidates, fluid_candidates, fluid_system) = system

    u_fluid = wrap_u(u_ode, fluid_system, semi)
    v_fluid = wrap_v(v_ode, fluid_system, semi)

    boundary_candidates .= false

    # Check the boundary particles whether they're leaving the boundary zone
    @threaded semi for particle in each_integrated_particle(system)
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
    @threaded semi for fluid_particle in each_integrated_particle(fluid_system)
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
@inline function convert_particle!(system::OpenBoundarySystem, fluid_system,
                                   boundary_zone, particle, particle_new,
                                   v, u, v_fluid, u_fluid)
    # Position relative to the origin of the transition face
    relative_position = current_coords(u, system, particle) - boundary_zone.zone_origin

    # Check if particle is in- or outside the fluid domain.
    # `face_normal` is always pointing into the fluid domain.
    # Since this function is called for a particle that left the boundary zone,
    # it is sufficient to check if the dot product between the relative position and the face normal is negative
    # to determine if it exited the boundary zone through the free surface (outflow).
    if dot(relative_position, boundary_zone.face_normal) < 0
        # Particle is outside the fluid domain
        deactivate_particle!(system, particle, u)

        return system
    end

    # Activate a new particle in simulation domain
    transfer_particle!(fluid_system, system, particle, particle_new, v_fluid, u_fluid, v, u)

    # Reset position of boundary particle
    for dim in 1:ndims(system)
        u[dim, particle] += boundary_zone.spanning_set[1][dim]
    end

    impose_rest_density!(v, system, particle, system.boundary_model)

    impose_rest_pressure!(v, system, particle, system.boundary_model)

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

function write_v0!(v0, system::OpenBoundarySystem)
    # This is as fast as a loop with `@inbounds`, but it's GPU-compatible
    indices = CartesianIndices(system.initial_condition.velocity)
    copyto!(v0, indices, system.initial_condition.velocity, indices)

    write_v0!(v0, system, system.boundary_model)

    return v0
end

write_v0!(v0, system::OpenBoundarySystem, boundary_model) = v0

function write_u0!(u0, system::OpenBoundarySystem)
    (; initial_condition) = system

    # This is as fast as a loop with `@inbounds`, but it's GPU-compatible
    indices = CartesianIndices(initial_condition.coordinates)
    copyto!(u0, indices, initial_condition.coordinates, indices)

    return u0
end

# To account for boundary effects in the viscosity term of the RHS, use the viscosity model
# of the neighboring particle systems.
@inline function viscosity_model(system::OpenBoundarySystem,
                                 neighbor_system::AbstractFluidSystem)
    return neighbor_system.viscosity
end

@inline function viscosity_model(system::OpenBoundarySystem,
                                 neighbor_system::AbstractBoundarySystem)
    return neighbor_system.boundary_model.viscosity
end

# When the neighbor is an `OpenBoundarySystem`, just use the viscosity of the `FluidSystem` instead
@inline function viscosity_model(system, neighbor_system::OpenBoundarySystem)
    return neighbor_system.fluid_system.viscosity
end

function system_data(system::OpenBoundarySystem, dv_ode, du_ode, v_ode, u_ode, semi)
    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    coordinates = current_coordinates(u, system)
    velocity = current_velocity(v, system)
    density = current_density(v, system)
    pressure = current_pressure(v, system)

    return (; coordinates, velocity, density, pressure)
end

function available_data(::OpenBoundarySystem)
    return (:coordinates, :velocity, :density, :pressure)
end

# One face of the boundary zone is the transition to the fluid domain.
# The face opposite to this transition face is a free surface.
@inline function modify_shifting_at_free_surfaces!(system::OpenBoundarySystem, u, semi)
    (; fluid_system, cache) = system

    @threaded semi for particle in each_integrated_particle(system)
        boundary_zone = current_boundary_zone(system, particle)

        particle_coords = current_coords(u, system, particle)

        # The zone origin lies within the transition plane between boundary zone and fluid domain
        dist_to_transition = dot(particle_coords - boundary_zone.zone_origin,
                                 -boundary_zone.face_normal)
        dist_free_surface = boundary_zone.zone_width - dist_to_transition

        if dist_free_surface < compact_support(fluid_system, fluid_system)
            # Disable shifting for this particle.
            # Note that Sun et al. 2017 propose a more sophisticated approach with a transition phase
            # where only the component orthogonal to the surface normal is kept and the tangential
            # component is set to zero. However, we assume laminar flow in the boundary zone,
            # so we simply disable shifting completely.
            for dim in 1:ndims(system)
                cache.delta_v[dim, particle] = zero(eltype(system))
            end
        end
    end

    return system
end

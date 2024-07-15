@doc raw"""
    OpenBoundarySPHSystem(boundary_zone::Union{InFlow, OutFlow}; sound_speed,
                          fluid_system::FluidSystem, buffer_size::Integer,
                          reference_velocity=zeros(ndims(boundary_zone)),
                          reference_pressure=0.0,
                          reference_density=first(boundary_zone.initial_condition.density))

Open boundary system for in- and outflow particles.
These open boundaries use the characteristic variables to propagate the appropriate values
to the outlet or inlet and have been proposed by Lastiwka et al. (2009). For more information
about the method see [description below](@ref method_of_characteristics).

# Arguments
- `boundary_zone`: Use [`InFlow`](@ref) for an inflow and [`OutFlow`](@ref) for an outflow boundary.

# Keywords
- `sound_speed`: Speed of sound.
- `fluid_system`: The corresponding fluid system
- `buffer_size`: Number of buffer particles.
- `reference_velocity`: Reference velocity is either a function mapping each particle's coordinates
                        and time to its velocity, an array where the ``i``-th column holds
                        the velocity of particle ``i`` or, for a constant fluid velocity,
                        a vector holding this velocity. Velocity is constant zero by default.
- `reference_pressure`: Reference pressure is either a function mapping each particle's coordinates
                        and time to its pressure, a vector holding the pressure of each particle,
                        or a scalar for a constant pressure over all particles.
                        Pressure is constant zero by default.
- `reference_density`: Reference density is either a function mapping each particle's coordinates
                       and time to its density, a vector holding the density of each particle,
                       or a scalar for a constant density over all particles.
                       Density is the density of the first particle in the initial condition by default.

!!! warning "Experimental Implementation"
	This is an experimental feature and may change in any future releases.
"""
struct OpenBoundarySPHSystem{BZ, NDIMS, ELTYPE <: Real, IC, FS, ARRAY1D, ARRAY2D, RV, RP,
                             RD, B} <: System{NDIMS, IC}
    initial_condition        :: IC
    fluid_system             :: FS
    smoothing_length         :: ELTYPE
    mass                     :: ARRAY1D # Array{ELTYPE, 1}: [particle]
    density                  :: ARRAY1D # Array{ELTYPE, 1}: [particle]
    volume                   :: ARRAY1D # Array{ELTYPE, 1}: [particle]
    pressure                 :: ARRAY1D # Array{ELTYPE, 1}: [particle]
    characteristics          :: ARRAY2D # Array{ELTYPE, 2}: [characteristic, particle]
    previous_characteristics :: ARRAY2D # Array{ELTYPE, 2}: [characteristic, particle]
    sound_speed              :: ELTYPE
    boundary_zone            :: BZ
    flow_direction           :: SVector{NDIMS, ELTYPE}
    reference_velocity       :: RV
    reference_pressure       :: RP
    reference_density        :: RD
    buffer                   :: B
    update_callback_used     :: Ref{Bool}

    function OpenBoundarySPHSystem(boundary_zone::Union{InFlow, OutFlow}; sound_speed,
                                   fluid_system::FluidSystem, buffer_size::Integer,
                                   reference_velocity=nothing,
                                   reference_pressure=nothing,
                                   reference_density=nothing)
        (; initial_condition) = boundary_zone

        buffer = SystemBuffer(nparticles(initial_condition), buffer_size)

        initial_condition = allocate_buffer(initial_condition, buffer)

        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)

        if !(reference_velocity isa Function || isnothing(reference_velocity) ||
             (reference_velocity isa Vector && length(reference_velocity) == NDIMS))
            throw(ArgumentError("`reference_velocity` must be either a function mapping " *
                                "each particle's coordinates and time to its velocity, " *
                                "an array where the ``i``-th column holds the velocity of particle ``i`` " *
                                "or, for a constant fluid velocity, a vector of length $NDIMS for a $(NDIMS)D problem holding this velocity"))
        else
            reference_velocity_ = wrap_reference_function(reference_velocity, Val(NDIMS))
        end

        if !(reference_pressure isa Function || reference_pressure isa Real ||
             isnothing(reference_pressure))
            throw(ArgumentError("`reference_pressure` must be either a function mapping " *
                                "each particle's coordinates and time to its pressure, " *
                                "a vector holding the pressure of each particle, or a scalar"))
        else
            reference_pressure_ = wrap_reference_function(reference_pressure, Val(NDIMS))
        end

        if !(reference_density isa Function || reference_density isa Real ||
             isnothing(reference_density))
            throw(ArgumentError("`reference_density` must be either a function mapping " *
                                "each particle's coordinates and time to its density, " *
                                "a vector holding the density of each particle, or a scalar"))
        else
            reference_density_ = wrap_reference_function(reference_density, Val(NDIMS))
        end

        if isnothing(reference_pressure)
            pressure = copy(initial_condition.pressure)
        else
            pressure = [reference_pressure_(initial_condition.coordinates[:, i], 0.0)
                        for i in eachparticle(initial_condition)]
        end
        mass = copy(initial_condition.mass)
        density = copy(initial_condition.density)
        volume = similar(initial_condition.density)

        characteristics = zeros(ELTYPE, 3, length(mass))
        previous_characteristics = zeros(ELTYPE, 3, length(mass))

        flow_direction_ = boundary_zone.flow_direction

        return new{typeof(boundary_zone), NDIMS, ELTYPE, typeof(initial_condition),
                   typeof(fluid_system), typeof(mass), typeof(characteristics),
                   typeof(reference_velocity_), typeof(reference_pressure_),
                   typeof(reference_density_),
                   typeof(buffer)}(initial_condition, fluid_system,
                                   fluid_system.smoothing_length, mass, density, volume,
                                   pressure, characteristics, previous_characteristics,
                                   sound_speed, boundary_zone, flow_direction_,
                                   reference_velocity_, reference_pressure_,
                                   reference_density_, buffer, false)
    end
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
    (; reference_velocity, reference_pressure, reference_density) = system
    v_ref = isnothing(reference_velocity(0, 0)) ? "-" : string(nameof(reference_velocity))
    p_ref = isnothing(reference_pressure(0, 0)) ? "-" : string(nameof(reference_pressure))
    rho_ref = isnothing(reference_density(0, 0)) ? "-" : string(nameof(reference_density))

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "OpenBoundarySPHSystem{$(ndims(system))}")
        summary_line(io, "#particles", nparticles(system))
        summary_line(io, "#buffer_particles", system.buffer.buffer_size)
        summary_line(io, "fluid system", type2string(system.fluid_system))
        summary_line(io, "boundary", type2string(system.boundary_zone))
        summary_line(io, "flow direction", system.flow_direction)
        summary_line(io, "prescribed velocity", v_ref)
        summary_line(io, "prescribed pressure", p_ref)
        summary_line(io, "prescribed density", rho_ref)
        summary_line(io, "width", round(system.boundary_zone.zone_width, digits=3))
        summary_footer(io)
    end
end

@inline function smoothing_kernel(system::OpenBoundarySPHSystem, distance)
    smoothing_kernel(system.fluid_system, distance)
end

function reset_callback_flag!(system::OpenBoundarySPHSystem)
    system.update_callback_used[] = false

    return system
end

update_callback_used!(system::OpenBoundarySPHSystem) = system.update_callback_used[] = true

@inline source_terms(system::OpenBoundarySPHSystem) = nothing

@inline hydrodynamic_mass(system::OpenBoundarySPHSystem, particle) = system.mass[particle]

@inline function particle_density(v, system::OpenBoundarySPHSystem, particle)
    return system.density[particle]
end

@inline function particle_pressure(v, system::OpenBoundarySPHSystem, particle)
    return system.pressure[particle]
end

@inline function update_quantities!(system::OpenBoundarySPHSystem, v, u, t)
    (; density, pressure, characteristics, flow_direction, sound_speed) = system

    # Update quantities based on the characteristic variables
    @threaded system for particle in each_moving_particle(system)
        particle_position = current_coords(u, system, particle)

        J1 = characteristics[1, particle]
        J2 = characteristics[2, particle]
        J3 = characteristics[3, particle]

        rho_ref, p_ref, v_ref = reference_values(v, system, particle, particle_position, t)

        density[particle] = rho_ref + ((-J1 + 0.5 * (J2 + J3)) / sound_speed^2)

        pressure[particle] = p_ref + 0.5 * (J2 + J3)

        rho = density[particle]
        v_ = v_ref + ((J2 - J3) / (2 * sound_speed * rho)) * flow_direction

        for dim in 1:ndims(system)
            v[dim, particle] = v_[dim]
        end
    end

    return system
end

function update_final!(system::OpenBoundarySPHSystem, v, u, v_ode, u_ode, semi, t;
                       update_from_callback=false)
    if !update_from_callback && !(system.update_callback_used[])
        throw(ArgumentError("`UpdateCallback` is required when using `OpenBoundarySPHSystem`"))
    end

    # interpolate_reference_values!(system, system.fluid_system, v, u, v_ode, u_ode, semi, t)

    @trixi_timeit timer() "evaluate characteristics" evaluate_characteristics!(system, v, u,
                                                                               v_ode, u_ode,
                                                                               semi, t)
end

# DualSPHysics (Tafuni et al.) interpolate the boundary values directly from the interior
# fluid domain to the boundary particles by mirroring the boundary particles into
# the fluid domain with ghost nodes.
# Thus, they have to do a taylor series approximation `f_0 = f_k + (r_0 - r_k) ⋅ ∇f_k`,
# where `∇f_k` is the corrected gradient calculated at the ghost nodes.
#
# The method of characteristic do not calculated the boundary values directly but evaluate
# the characteristic variables with reference values `rho_ref`, `v_ref` and `p_ref`.
# Thus, it should be possible to interpolate the `ref` values directly at the ghost node position
# without any corrections.
#
# Tafuni et al. (2018)
# "A versatile algorithm for the treatment of open boundary conditions in Smoothed particle hydrodynamics GPU models"
# https://doi.org/10.1016/j.cma.2018.08.004
#
function interpolate_reference_values!(system, fluid_system::FluidSystem,
                                       v, u, v_ode, u_ode, semi, t)
    (; volume, initial_condition, boundary_zone) = system
    (; density, pressure, velocity) = initial_condition

    set_zero!(volume)
    set_zero!(density)
    set_zero!(pressure)
    set_zero!(velocity)

    v_fluid_system = wrap_v(v_ode, fluid_system, semi)
    u_fluid_system = wrap_u(u_ode, fluid_system, semi)

    fluid_coords = current_coordinates(u_fluid_system, fluid_system)
    nhs = get_neighborhood_search(system, fluid_system, semi)

    for particle in each_moving_particle(system)
        particle_coords = current_coords(u, system, particle)

        ghost_node_position = mirror_position(particle_coords, boundary_zone)

        # Check the neighboring fluid particles whether they're entering the boundary zone
        for neighbor in PointNeighbors.eachneighbor(ghost_node_position, nhs)
            fluid_coords = current_coords(u_fluid_system, fluid_system, neighbor)

            pos_diff = ghost_node_position - fluid_coords
            distance2 = dot(pos_diff, pos_diff)
            if distance2 <= nhs.search_radius^2
                distance = sqrt(distance2)
                rho_b = particle_density(v_fluid_system, fluid_system, neighbor)

                p_b = particle_pressure(v_fluid_system, fluid_system, neighbor)

                v_fluid = current_velocity(v_fluid_system, fluid_system, neighbor)

                kernel_weight = smoothing_kernel(fluid_system, distance)

                pressure[particle] += p_b * kernel_weight

                density[particle] += rho_b * kernel_weight

                for dim in 1:ndims(system)
                    velocity[dim, particle] += v_fluid[dim] * kernel_weight
                end

                volume[particle] += kernel_weight
            end
        end

    end

    for particle in each_moving_particle(system)
        if volume[particle] > eps()
            pressure[particle] /= volume[particle]
            density[particle] /= volume[particle]
            for dim in 1:ndims(system)
                velocity[dim, particle] /= volume[particle]
            end
        end
    end

    return system
end

# ==== Characteristics
# J1: Associated with convection and entropy and propagates at flow velocity.
# J2: Propagates downstream to the local flow
# J3: Propagates upstream to the local flow
function evaluate_characteristics!(system, v, u, v_ode, u_ode, semi, t)
    (; volume, characteristics, previous_characteristics, boundary_zone) = system

    for particle in eachparticle(system)
        previous_characteristics[1, particle] = characteristics[1, particle]
        previous_characteristics[2, particle] = characteristics[2, particle]
        previous_characteristics[3, particle] = characteristics[3, particle]
    end

    set_zero!(characteristics)
    set_zero!(volume)

    # Evaluate the characteristic variables with the fluid system
    evaluate_characteristics!(system, system.fluid_system, v, u, v_ode, u_ode, semi, t)

    # Only some of the in-/outlet particles are in the influence of the fluid particles.
    # Thus, we compute the characteristics for the particles that are outside the influence
    # of fluid particles by using the average of the values of the previous time step.
    # See eq. 27 in Negi (2020) https://doi.org/10.1016/j.cma.2020.113119
    @threaded system for particle in each_moving_particle(system)

        # Particle is outside of the influence of fluid particles
        if isapprox(volume[particle], 0.0)

            # Using the average of the values at the previous time step for particles which
            # are outside of the influence of fluid particles.
            avg_J1 = 0.0
            avg_J2 = 0.0
            avg_J3 = 0.0
            counter = 0

            for neighbor in each_moving_particle(system)
                # Make sure that only neighbors in the influence of
                # the fluid particles are used.
                if volume[neighbor] > sqrt(eps())
                    avg_J1 += previous_characteristics[1, neighbor]
                    avg_J2 += previous_characteristics[2, neighbor]
                    avg_J3 += previous_characteristics[3, neighbor]
                    counter += 1
                end
            end

            characteristics[1, particle] = avg_J1 / counter
            characteristics[2, particle] = avg_J2 / counter
            characteristics[3, particle] = avg_J3 / counter
        else
            characteristics[1, particle] /= volume[particle]
            characteristics[2, particle] /= volume[particle]
            characteristics[3, particle] /= volume[particle]
        end
        prescribe_conditions!(characteristics, particle, boundary_zone)
    end

    return system
end

function evaluate_characteristics!(system, neighbor_system::FluidSystem,
                                   v, u, v_ode, u_ode, semi, t)
    (; volume, sound_speed, characteristics, flow_direction) = system

    v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)
    u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)

    nhs = get_neighborhood_search(system, neighbor_system, semi)

    system_coords = current_coordinates(u, system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # Loop over all fluid neighbors within the kernel cutoff
    foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords,
                           nhs) do particle, neighbor, pos_diff, distance
        neighbor_position = current_coords(u_neighbor_system, neighbor_system, neighbor)

        # Determine current and prescribed quantities
        rho_b = particle_density(v_neighbor_system, neighbor_system, neighbor)

        p_b = particle_pressure(v_neighbor_system, neighbor_system, neighbor)

        v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)

        rho_ref, p_ref, v_ref = reference_values(v, system, particle, neighbor_position, t)

        # Determine characteristic variables
        density_term = -sound_speed^2 * (rho_b - rho_ref)
        pressure_term = p_b - p_ref
        velocity_term = rho_b * sound_speed * (dot(v_b - v_ref, flow_direction))

        kernel_ = smoothing_kernel(neighbor_system, distance)

        characteristics[1, particle] += (density_term + pressure_term) * kernel_
        characteristics[2, particle] += (velocity_term + pressure_term) * kernel_
        characteristics[3, particle] += (-velocity_term + pressure_term) * kernel_

        volume[particle] += kernel_
    end

    return system
end

@inline function prescribe_conditions!(characteristics, particle, ::OutFlow)
    # J3 is prescribed (i.e. determined from the exterior of the domain).
    # J1 and J2 is transimtted from the domain interior.
    characteristics[3, particle] = zero(eltype(characteristics))

    return characteristics
end

@inline function prescribe_conditions!(characteristics, particle, ::InFlow)
    # Allow only J3 to propagate upstream to the boundary
    characteristics[1, particle] = zero(eltype(characteristics))
    characteristics[2, particle] = zero(eltype(characteristics))

    return characteristics
end

# This function is called by the `UpdateCallback`, as the integrator array might be modified
function update_open_boundary_eachstep!(system::OpenBoundarySPHSystem, v_ode, u_ode,
                                        semi, t)
    u = wrap_u(u_ode, system, semi)
    v = wrap_v(v_ode, system, semi)

    # Update density, pressure and velocity based on the characteristic variables.
    # See eq. 13-15 in Lastiwka (2009) https://doi.org/10.1002/fld.1971
    @trixi_timeit timer() "update quantities" update_quantities!(system, v, u, t)

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

    # TODO: Only when using TVF: set tvf

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

function reference_values(v, system, particle, position, t)
    (; reference_velocity, reference_pressure, reference_density,
    initial_condition) = system

    rho_ref = reference_density(position, t)
    p_ref = reference_pressure(position, t)
    v_ref = reference_velocity(position, t)

    if isnothing(rho_ref)
        rho_ref = initial_condition.density[particle]
    end
    if isnothing(p_ref)
        p_ref = initial_condition.pressure[particle]
    end
    if isnothing(v_ref)
        v_ref = current_velocity(v, system, particle)
    end

    return rho_ref, p_ref, v_ref
end

function mirror_position(particle_coords, boundary_zone)
    particle_position = particle_coords - boundary_zone.zone_origin
    dist = dot(particle_position, boundary_zone.flow_direction)

    return particle_coords - 2 * dist * boundary_zone.flow_direction
end

function wrap_reference_function(::Nothing, ::Val)
    return (coords, t) -> nothing
end

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

# To account for boundary effects in the viscosity term of the RHS, use the viscosity model
# of the neighboring particle systems.
@inline viscosity_model(system::OpenBoundarySPHSystem, neighbor_system::FluidSystem) = neighbor_system.viscosity
@inline viscosity_model(system::OpenBoundarySPHSystem, neighbor_system::BoundarySystem) = neighbor_system.boundary_model.viscosity
# When the neighbor is an open boundary system, just use the viscosity of the fluid `system` instead
@inline viscosity_model(system, neighbor_system::OpenBoundarySPHSystem) = system.viscosity

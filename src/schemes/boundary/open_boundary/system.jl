"""
    InFlow

Inflow boundary zone for [`OpenBoundarySPHSystem`](@ref)
"""
struct InFlow end

"""
    OutFlow

Outflow boundary zone for [`OpenBoundarySPHSystem`](@ref)
"""
struct OutFlow end

"""
    OpenBoundarySPHSystem(plane_points, boundary_zone::Union{InFlow, OutFlow},
                          sound_speed;
                          sample_geometry=plane_points, particle_spacing,
                          flow_direction, open_boundary_layers::Integer=0, density,
                          buffer=nothing, reference_velocity=zero(flow_direction),
                          reference_pressure=0.0, reference_density=density)
Open boundary system for in- and outflow particles.
These open boundaries use the characteristic variables to propagate the appropriate values
to the outlet or inlet and has been proposed by Lastiwka et al (2009). For more information
about the method see [Open Boundary System](@ref open_boundary).

# Arguments
- `plane_points`: Points defining the boundary zones front plane.
                  The points must either span a rectangular plane in 3D or a line in 2D.
                  See description above for more information.
- `boundary_zone`: Use [`InFlow`](@ref) for an inflow and [`OutFlow`](@ref) for an outflow boundary.
- `sound_speed`: Speed of sound.

# Keywords
- `sample_plane`: For customized particle sampling in the boundary zone, this can be either
                  points defining a 3D plane (2D line), particle coordinates defining a specific
                  shape or a specific [`InitialCondition`](@ref) type.
                  The geometry will be extruded in upstream direction with [`ExtrudeGeometry`](@ref).
                  Default is `plane_points` which fully samples the boundary zone with particles.
- `particle_spacing`: The spacing between the particles in the boundary zone.
- `flow_direction`: Vector defining the flow direction.
- `open_boundary_layers`: Number of particle layers in upstream direction.
- `density`: Density of each particle to define the mass of each particle (see [`InitialCondition`](@ref)).
- `buffer`: Number of buffer particles.
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
                       Density is constant zero by default.

# Examples
```julia
# 2D inflow
plane_points = ([0.0, 0.0], [0.0, 1.0])
flow_direction=[1.0, 0.0]

system = OpenBoundarySPHSystem(plane_points, InFlow(), 10.0; particle_spacing=0.1,
                               open_boundary_layers=4, density=1.0, flow_direction)

# 3D outflow
plane_points = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
flow_direction=[0.0, 0.0, 1.0]

system = OpenBoundarySPHSystem(plane_points, OutFlow(), 10.0; particle_spacing=0.1,
                               open_boundary_layers=4, density=1.0, flow_direction)

# 3D particles sampled as cylinder
circle = SphereShape(0.1, 0.5, (0.5, 0.5), 1.0, sphere_type=RoundSphere())

system = OpenBoundarySPHSystem(plane_points, InFlow(), 10.0; particle_spacing=0.1,
                               sample_geometry=circle,
                               open_boundary_layers=4, density=1.0, flow_direction)
```
"""
struct OpenBoundarySPHSystem{BZ, NDIMS, ELTYPE <: Real, S, RV, RP, RD, B} <: System{NDIMS}
    initial_condition        :: InitialCondition{ELTYPE}
    mass                     :: Array{ELTYPE, 1} # [particle]
    density                  :: Array{ELTYPE, 1} # [particle]
    volume                   :: Array{ELTYPE, 1} # [particle]
    pressure                 :: Array{ELTYPE, 1} # [particle]
    characteristics          :: Array{ELTYPE, 2} # [characteristics, particle]
    previous_characteristics :: Array{ELTYPE, 2} # [characteristics, particle]
    sound_speed              :: ELTYPE
    boundary_zone            :: BZ
    flow_direction           :: SVector{NDIMS, ELTYPE}
    zone_origin              :: SVector{NDIMS, ELTYPE}
    spanning_set             :: S
    reference_velocity       :: RV
    reference_pressure       :: RP
    reference_density        :: RD
    buffer                   :: B

    function OpenBoundarySPHSystem(plane_points, boundary_zone::Union{InFlow, OutFlow},
                                   sound_speed;
                                   sample_geometry=plane_points, particle_spacing,
                                   flow_direction, open_boundary_layers::Integer=0, density,
                                   buffer=nothing,
                                   reference_velocity=zeros(length(flow_direction)),
                                   reference_pressure=0.0, reference_density=density)
        if open_boundary_layers < sqrt(eps())
            throw(ArgumentError("`open_boundary_layers` must be positive and greater than zero"))
        end

        # Unit vector pointing in downstream direction.
        flow_direction_ = normalize(SVector(flow_direction...))

        # Sample boundary zone with particles in corresponding direction.
        (boundary_zone isa OutFlow) && (direction = flow_direction_)
        (boundary_zone isa InFlow) && (direction = -flow_direction_)

        # Sample particles in boundary zone.
        initial_condition = ExtrudeGeometry(sample_geometry; particle_spacing, direction,
                                            n_extrude=open_boundary_layers, density)

        (buffer ≠ nothing) && (buffer = SystemBuffer(nparticles(initial_condition), buffer))
        initial_condition = allocate_buffer(initial_condition, buffer)

        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)

        if !(reference_velocity isa Function ||
             (reference_velocity isa Vector && length(reference_velocity) == NDIMS))
            throw(ArgumentError("`reference_velocity` must be either a function mapping " *
                                "each particle's coordinates and time to its velocity or a " *
                                "vector of length $NDIMS for a $(NDIMS)D problem"))
        else
            reference_velocity_ = wrap_reference_function(reference_velocity, Val(NDIMS))
        end

        if !(reference_pressure isa Function || reference_pressure isa Real)
            throw(ArgumentError("`reference_pressure` must be either a function mapping " *
                                "each particle's coordinates and time to its pressure or a scalar"))
        else
            reference_pressure_ = wrap_reference_function(reference_pressure, Val(NDIMS))
        end

        if !(reference_density isa Function || reference_density isa Real)
            throw(ArgumentError("`reference_density` must be either a function mapping " *
                                "each particle's coordinates and time to its density or a scalar"))
        else
            reference_density_ = wrap_reference_function(reference_density, Val(NDIMS))
        end

        zone_width = open_boundary_layers * initial_condition.particle_spacing

        # Vectors spanning the boundary zone/box.
        spanning_set = spanning_vectors(plane_points, zone_width)

        # First vector of `spanning_vectors` is normal to the in-/outflow plane.
        # The normal vector must point in downstream direction for an outflow boundary and
        # for an inflow boundary the normal vector must point in upstream direction.
        # Thus, rotate the normal vector correspondingly.
        if isapprox(dot(normalize(spanning_set[:, 1]), flow_direction_), 1.0, atol=1e-7)
            # Normal vector points in downstream direction.
            # Flip the inflow vector in upstream direction
            (boundary_zone isa InFlow) && (spanning_set[:, 1] .*= -1)
        elseif isapprox(dot(normalize(spanning_set[:, 1]), flow_direction_), -1.0,
                        atol=1e-7)
            # Normal vector points in upstream direction.
            # Flip the outflow vector in downstream direction
            (boundary_zone isa OutFlow) && (spanning_set[:, 1] .*= -1)
        else
            throw(ArgumentError("flow direction and normal vector of " *
                                "$(typeof(boundary_zone))-plane do not correspond"))
        end

        spanning_set_ = reinterpret(reshape, SVector{NDIMS, ELTYPE}, spanning_set)

        zone_origin = SVector(plane_points[1]...)

        mass = copy(initial_condition.mass)
        pressure = [reference_pressure_(initial_condition.coordinates[:, i], 0.0)
                    for i in eachparticle(initial_condition)]
        density = copy(initial_condition.density)
        volume = similar(initial_condition.density)

        characteristics = zeros(ELTYPE, 3, length(mass))
        previous_characteristics = zeros(ELTYPE, 3, length(mass))

        return new{typeof(boundary_zone), NDIMS, ELTYPE, typeof(spanning_set_),
                   typeof(reference_velocity_), typeof(reference_pressure_),
                   typeof(reference_density_),
                   typeof(buffer)}(initial_condition, mass, density, volume, pressure,
                                   characteristics, previous_characteristics, sound_speed,
                                   boundary_zone, flow_direction_, zone_origin,
                                   spanning_set_, reference_velocity_, reference_pressure_,
                                   reference_density_, buffer)
    end
end

timer_name(::OpenBoundarySPHSystem) = "open_boundary"
vtkname(system::OpenBoundarySPHSystem) = "open_boundary"

function Base.show(io::IO, system::OpenBoundarySPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "OpenBoundarySPHSystem{", ndims(system), "}(")
    print(io, system.boundary_zone)
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::OpenBoundarySPHSystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "OpenBoundarySPHSystem{$(ndims(system))}")
        if system.buffer isa SystemBuffer
            summary_line(io, "#particles", nparticles(system))
            summary_line(io, "#buffer_particles", system.buffer.buffer_size)
        else
            summary_line(io, "#particles", nparticles(system))
        end
        summary_line(io, "boundary", system.boundary_zone)
        summary_line(io, "flow direction", system.flow_direction)
        summary_line(io, "prescribed velocity", string(nameof(system.reference_velocity)))
        summary_line(io, "prescribed pressure", string(nameof(system.reference_pressure)))
        summary_line(io, "prescribed density", string(nameof(system.reference_density)))
        summary_line(io, "width", round(norm(system.spanning_set[1]), digits=3))
        summary_footer(io)
    end
end

@inline source_terms(system::OpenBoundarySPHSystem) = nothing

@inline hydrodynamic_mass(system::OpenBoundarySPHSystem, particle) = system.mass[particle]

@inline function particle_density(v, system::OpenBoundarySPHSystem, particle)
    return system.density[particle]
end

@inline function particle_pressure(v, system::OpenBoundarySPHSystem, particle)
    return system.pressure[particle]
end

function spanning_vectors(plane_points, zone_width)

    # Convert to tuple
    return spanning_vectors(tuple(plane_points...), zone_width)
end

function spanning_vectors(plane_points::NTuple{2}, zone_width)
    plane_size = plane_points[2] - plane_points[1]

    # Calculate normal vector of plane
    b = Vector(normalize([-plane_size[2]; plane_size[1]]) * zone_width)

    return hcat(b, plane_size)
end

function spanning_vectors(plane_points::NTuple{3}, zone_width)
    # Vectors spanning the plane
    edge1 = plane_points[2] - plane_points[1]
    edge2 = plane_points[3] - plane_points[1]

    if !isapprox(dot(edge1, edge2), 0.0, atol=1e-7)
        throw(ArgumentError("the provided points do not span a rectangular plane"))
    end

    # Calculate normal vector of plane
    c = Vector(normalize(cross(edge2, edge1)) * zone_width)

    return hcat(c, edge1, edge2)
end

@inline function within_boundary_zone(particle_coords, system)
    (; zone_origin, spanning_set) = system
    particle_position = particle_coords - zone_origin

    for dim in 1:ndims(system)
        span_dim = spanning_set[dim]
        # Checks whether the projection of the particle position
        # falls within the range of the zone.
        if !(0 <= dot(particle_position, span_dim) <= dot(span_dim, span_dim))

            # Particle is not in boundary zone.
            return false
        end
    end

    # Particle is in boundary zone.
    return true
end

function update_final!(system::OpenBoundarySPHSystem, v, u, v_ode, u_ode, semi, t)
    @trixi_timeit timer() "evaluate characteristics" evaluate_characteristics!(system, v, u,
                                                                               v_ode, u_ode,
                                                                               semi, t)
end

update_open_boundary_eachstep!(system, v_ode, u_ode, semi, t) = system

function update_open_boundary_eachstep!(system::OpenBoundarySPHSystem, v_ode, u_ode,
                                        semi, t)
    u = wrap_u(u_ode, system, semi)
    v = wrap_v(v_ode, system, semi)

    @trixi_timeit timer() "update quantities" update_quantities!(system, v, u, t)

    @trixi_timeit timer() "check domain" check_domain!(system, v, u, v_ode, u_ode, semi)

    @trixi_timeit timer() "update buffer" foreach_system(semi) do system
        update_system_buffer!(system.buffer)
    end
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

    # Use all other systems for the characteristics
    foreach_system(semi) do neighbor_system
        evaluate_characteristics!(system, neighbor_system, v, u, v_ode, u_ode, semi, t)
    end

    # Only some of the in-/outlet particles are in the influence of the fluid particles.
    # Thus, we find the characteristics for the particle which are outside the influence
    # using the average of the values of the previous time step.
    # Negi (2020) https://doi.org/10.1016/j.cma.2020.113119
    @threaded for particle in each_moving_particle(system)

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

evaluate_characteristics!(system, neighbor_system, v, u, v_ode, u_ode, semi, t) = system

function evaluate_characteristics!(system, neighbor_system::FluidSystem,
                                   v, u, v_ode, u_ode, semi, t)
    (; volume, sound_speed, characteristics, flow_direction,
    reference_velocity, reference_pressure, reference_density) = system

    v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)
    u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)

    nhs = get_neighborhood_search(system, neighbor_system, semi)

    system_coords = current_coordinates(u, system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # Loop over all fluid neighbors within the kernel cutoff.
    for_particle_neighbor(system, neighbor_system, system_coords, neighbor_coords,
                          nhs) do particle, neighbor, pos_diff, distance
        neighbor_position = current_coords(u_neighbor_system, neighbor_system, neighbor)

        # Determine current and prescribed quantities
        rho = particle_density(v_neighbor_system, neighbor_system, neighbor)
        rho_ref = reference_density(neighbor_position, t)

        p = particle_pressure(v_neighbor_system, neighbor_system, neighbor)
        p_ref = reference_pressure(neighbor_position, t)

        v_neighbor = current_velocity(v_neighbor_system, neighbor_system, neighbor)
        v_neighbor_ref = reference_velocity(neighbor_position, t)

        # Determine characteristic variables
        density_term = -sound_speed^2 * (rho - rho_ref)
        pressure_term = p - p_ref
        velocity_term = rho * sound_speed *
                        (dot(v_neighbor - v_neighbor_ref, flow_direction))

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

@inline function update_quantities!(system::OpenBoundarySPHSystem, v, u, t)
    (; density, pressure, characteristics, flow_direction, sound_speed,
    reference_velocity, reference_pressure, reference_density) = system

    @threaded for particle in each_moving_particle(system)
        particle_position = current_coords(u, system, particle)

        J1 = characteristics[1, particle]
        J2 = characteristics[2, particle]
        J3 = characteristics[3, particle]

        rho_ref = reference_density(particle_position, t)
        density[particle] = rho_ref + ((-J1 + 0.5 * (J2 + J3)) / sound_speed^2)

        p_ref = reference_pressure(particle_position, t)
        pressure[particle] = p_ref + 0.5 * (J2 + J3)

        v_ref = reference_velocity(particle_position, t)
        rho = density[particle]
        v_ = v_ref + ((J2 - J3) / (2 * sound_speed * rho)) * flow_direction

        for dim in 1:ndims(system)
            v[dim, particle] = v_[dim]
        end
    end

    return system
end

function check_domain!(system, v, u, v_ode, u_ode, semi)
    # TODO: Is a thread supported version possible?
    for particle in each_moving_particle(system)
        foreach_system(semi) do fluid_system
            check_fluid_domain!(system, fluid_system, particle, v, u, v_ode, u_ode, semi)
        end
    end
end

function check_fluid_domain!(system, neighbor_system, particle, v, u, v_ode, u_ode, semi)
    return system
end

function check_fluid_domain!(system, fluid_system::FluidSystem, particle,
                             v, u, v_ode, u_ode, semi)
    (; boundary_zone) = system

    particle_coords = current_coords(u, system, particle)

    u_fluid = wrap_u(u_ode, fluid_system, semi)
    v_fluid = wrap_v(v_ode, fluid_system, semi)

    neighborhood_search = get_neighborhood_search(system, fluid_system, semi)

    # Check if the particle position is outside the boundary zone.
    if !within_boundary_zone(particle_coords, system)
        transform_particle!(system, fluid_system, boundary_zone, particle,
                            v, u, v_fluid, u_fluid)
    end

    # Check fluid neighbors
    for neighbor in eachneighbor(particle_coords, neighborhood_search)
        fluid_coords = current_coords(u_fluid, fluid_system, neighbor)

        # Check if neighbor position is in boundary zone
        if within_boundary_zone(fluid_coords, system)
            transform_particle!(fluid_system, system, boundary_zone, neighbor,
                                v, u, v_fluid, u_fluid)
        end
    end

    return system
end

# Outflow particle is outside the boundary zone
@inline function transform_particle!(system::OpenBoundarySPHSystem, fluid_system,
                                     ::OutFlow, particle, v, u, v_fluid, u_fluid)
    deactivate_particle!(system, particle, u)

    return system
end

# Inflow particle is outside the boundary zone
@inline function transform_particle!(system::OpenBoundarySPHSystem, fluid_system,
                                     ::InFlow, particle, v, u, v_fluid, u_fluid)
    (; spanning_set) = system

    # Activate a new particle in simulation domain
    activate_particle!(fluid_system, system, particle, v_fluid, u_fluid, v, u)

    # Reset position of boundary particle
    for dim in 1:ndims(system)
        u[dim, particle] += spanning_set[1][dim]
    end

    return system
end

# Fluid particle is in boundary zone
@inline function transform_particle!(fluid_system::FluidSystem, system,
                                     boundary_zone, particle, v, u, v_fluid, u_fluid)
    # Activate particle in boundary zone
    activate_particle!(system, fluid_system, particle, v, u, v_fluid, u_fluid)

    # Deactivate particle in interior domain
    deactivate_particle!(fluid_system, particle, u_fluid)

    return fluid_system
end

@inline function activate_particle!(system_new, system_old, particle_old,
                                    v_new, u_new, v_old, u_old)
    particle_new = available_particle(system_new)

    # Exchange densities
    density = particle_density(v_old, system_old, particle_old)
    set_particle_density(particle_new, v_new, system_new, density)

    # Exchange pressure
    pressure = particle_pressure(v_old, system_old, particle_old)
    set_particle_pressure(particle_new, v_new, system_new, pressure)

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

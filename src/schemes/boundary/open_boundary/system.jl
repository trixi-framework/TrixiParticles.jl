struct InFlow end

struct OutFlow end

struct OpenBoundarySPHSystem{BZ, NDIMS, ELTYPE <: Real, S, VF, PF, DF} <: System{NDIMS}
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
    velocity_function        :: VF
    pressure_function        :: PF
    density_function         :: DF

    function OpenBoundarySPHSystem(plane_points, boundary_zone, sound_speed;
                                   sample_geometry=plane_points, particle_spacing,
                                   flow_direction, open_boundary_layers=0, density,
                                   velocity=zeros(length(plane_points)), mass=nothing,
                                   pressure=0.0, velocity_function=nothing,
                                   pressure_function=nothing, density_function=nothing)
        if !((boundary_zone isa InFlow) || (boundary_zone isa OutFlow))
            throw(ArgumentError("`boundary_zone` must either be of type InFlow or OutFlow"))
        end

        if !(open_boundary_layers isa Int)
            throw(ArgumentError("`open_boundary_layers` must be of type Int"))
        elseif open_boundary_layers < sqrt(eps())
            throw(ArgumentError("`open_boundary_layers` must be positive and greater than zero"))
        end

        # Unit vector pointing in downstream direction.
        flow_direction_ = normalize(SVector(flow_direction...))

        # Sample boundary zone with particles in corresponding direction.
        (boundary_zone isa OutFlow) && (direction = flow_direction_)
        (boundary_zone isa InFlow) && (direction = -flow_direction_)

        # Sample particles in boundary zone.
        initial_condition = ExtrudeGeometry(sample_geometry; particle_spacing, direction,
                                            n_extrude=open_boundary_layers, velocity, mass,
                                            density, pressure)

        zone_width = open_boundary_layers * initial_condition.particle_spacing

        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)

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
        pressure = copy(initial_condition.pressure)
        density = copy(initial_condition.density)
        volume = similar(initial_condition.density)

        characteristics = zeros(ELTYPE, 3, length(mass))
        previous_characteristics = zeros(ELTYPE, 3, length(mass))

        return new{typeof(boundary_zone), NDIMS, ELTYPE, typeof(spanning_set_),
                   typeof(velocity_function), typeof(pressure_function),
                   typeof(density_function)}(initial_condition, mass, density, volume,
                                             pressure, characteristics,
                                             previous_characteristics, sound_speed,
                                             boundary_zone, flow_direction_, zone_origin,
                                             spanning_set_, velocity_function,
                                             pressure_function, density_function)
    end
end

timer_name(::OpenBoundarySPHSystem) = "open_boundary"

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
        summary_line(io, "#particles", nparticles(system))
        summary_line(io, "boundary", system.boundary_zone)
        summary_line(io, "flow direction", system.flow_direction)
        summary_line(io, "width", round(norm(system.spanning_set[1]), digits=3))
        summary_footer(io)
    end
end

@inline viscosity_model(system, neighbor_system::OpenBoundarySPHSystem) = system.viscosity

@inline source_terms(system::OpenBoundarySPHSystem) = nothing

@inline hydrodynamic_mass(system::OpenBoundarySPHSystem, particle) = system.mass[particle]

@inline function particle_density(v, system::OpenBoundarySPHSystem, particle)
    return system.density[particle]
end

@inline function particle_pressure(v, system::OpenBoundarySPHSystem, particle)
    return system.pressure[particle]
end

@inline set_particle_density(particle, v, system::OpenBoundarySPHSystem, density) = system

@inline set_particle_pressure(particle, v, system, pressure) = system

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
    particle_positon = particle_coords - zone_origin

    for dim in 1:ndims(system)
        span_dim = spanning_set[dim]
        # Checks whether the projection of the particle position
        # falls within the range of the zone.
        if !(0 <= dot(particle_positon, span_dim) <= dot(span_dim, span_dim))

            # Particle is not in boundary zone.
            return false
        end
    end

    # Particle is in boundary zone.
    return true
end

function update_final!(system::OpenBoundarySPHSystem, v, u, v_ode, u_ode, semi, t)
    evaluate_characteristics!(system, v, u, v_ode, u_ode, semi, t)
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
    @trixi_timeit timer() "Evaluate Characteristics" foreach_system(semi) do neighbor_system
        evaluate_characteristics!(system, neighbor_system, v, u, v_ode, u_ode, semi, t)
    end

    # Only some of the in-/outlet particles are in the influence of the interior particles.
    # Thus, we find the characteristics for the particle which are outside the influence
    # using the average of the values of the previous time step.
    @threaded for particle in each_moving_particle(system)

        # Particle is outside of the influence of interior particles
        if isapprox(volume[particle], 0.0)

            # Using the average of the values at the previous time step for particles which
            # are outside of the influence of interior particles.
            avg_J1 = 0.0
            avg_J2 = 0.0
            avg_J3 = 0.0
            counter = 0

            for neighbor in each_moving_particle(system)
                # Make sure that only neighbors in the influence of
                # the interior particles are used.
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
    (; volume, sound_speed, characteristics,
    velocity_function, density_function, pressure_function, flow_direction) = system

    v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)
    u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)

    nhs = get_neighborhood_search(system, neighbor_system, semi)

    system_coords = current_coordinates(u, system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # Loop over all interior neighbors within the kernel cutoff.
    for_particle_neighbor(system, neighbor_system, system_coords, neighbor_coords,
                          nhs) do particle, neighbor, pos_diff, distance
        # Determine current and prescribed quantities
        rho = particle_density(v_neighbor_system, neighbor_system, neighbor)
        rho_ref = reference_density(system, density_function,
                                    neighbor, u_neighbor_system, t)

        p = particle_pressure(v_neighbor_system, neighbor_system, neighbor)
        p_ref = reference_pressure(system, pressure_function,
                                   neighbor, u_neighbor_system, t)

        v_neighbor = current_velocity(v_neighbor_system, neighbor_system, neighbor)
        v_neighbor_ref = reference_velocity(system, velocity_function,
                                            neighbor, u_neighbor_system, t)

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

function reference_velocity(system, velocity_function, particle, u, t)
    position = current_coords(u, system, particle)
    return SVector{ndims(system)}(velocity_function(position, t))
end

function reference_velocity(system, ::Nothing, particle, u, t)
    return extract_svector(system.initial_condition.velocity, system, particle)
end

function reference_pressure(system, pressure_function, particle, u, t)
    position = current_coords(u, system, particle)
    return pressure_function(position, t)
end

function reference_pressure(system, ::Nothing, particle, u, t)
    return system.initial_condition.pressure[particle]
end

function reference_density(system, density_function, particle, u, t)
    position = current_coords(u, system, particle)
    return density_function(position, t)
end

function reference_density(system, ::Nothing, particle, u, t)
    return system.initial_condition.density[particle]
end

struct OpenBoundarySPHSystem{BZ, NDIMS, ELTYPE <: Real, V, B, VF} <: FluidSystem{NDIMS}
    initial_condition        :: InitialCondition{ELTYPE}
    mass                     :: Array{ELTYPE, 1} # [particle]
    volume                   :: Array{ELTYPE, 1} # [particle]
    density                  :: Array{ELTYPE, 1} # [particle]
    pressure                 :: Array{ELTYPE, 1} # [particle]
    characteristics          :: Array{ELTYPE, 2} # [characteristics, particle]
    previous_characteristics :: Array{ELTYPE, 2} # [characteristics, particle]
    sound_speed              :: ELTYPE
    boundary_zone            :: BZ
    interior_system          :: System
    zone_origin              :: SVector{NDIMS, ELTYPE}
    spanning_set             :: SMatrix{NDIMS, NDIMS, ELTYPE}
    flow_direction           :: SVector{NDIMS, ELTYPE}
    viscosity                :: V
    buffer                   :: B
    velocity_function        :: VF
    acceleration             :: SVector{NDIMS, ELTYPE}

    # TODO: Mutltiple interior systems. Same as in `semidiscretization(systems...)`
    function OpenBoundarySPHSystem(initial_condition, boundary_zone, interior_system;
                                   zone_width=0.0,
                                   flow_direction=ntuple(_ -> 0.0, ndims(interior_system)),
                                   zone_plane_min_corner=ntuple(_ -> 0.0,
                                                                ndims(interior_system)),
                                   zone_plane_max_corner=ntuple(_ -> 0.0,
                                                                ndims(interior_system)),
                                   velocity_function=nothing, buffer=nothing)
        (buffer ≠ nothing) && (buffer = SystemBuffer(nparticles(initial_condition), buffer))
        initial_condition = allocate_buffer(initial_condition, buffer)

        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)

        mass = copy(initial_condition.mass)
        pressure = copy(initial_condition.pressure)
        density = copy(initial_condition.density)
        volume = similar(initial_condition.density)

        viscosity = interior_system.viscosity
        sound_speed = speed_of_sound(interior_system)

        characteristics = zeros(ELTYPE, 3, length(mass))
        previous_characteristics = zeros(ELTYPE, 4, length(mass))

        zone_origin = SVector(zone_plane_min_corner...)

        # Unit vector pointing in downstream direction.
        flow_direction_ = normalize(SVector(flow_direction...))

        plane_size = SVector{NDIMS}(zone_plane_max_corner - zone_plane_min_corner)

        # Vectors spanning the boundary zone/box.
        spanning_set = spanning_vectors(plane_size, zone_width)

        # First vector of `spanning_vectors` is normal to the in-/outflow plane.
        # The normal vector must point in downstream direction for an outflow boundary and
        # for an inflow boundary the normal vector must point in upstream direction.
        # Thus, rotate the normal vector correspondingly.
        span_angle = 0
        if isapprox(dot(normalize(spanning_set[:, 1]), flow_direction_), 1.0)
            # Normal vector points in downstream direction.
            # Flip the inflow vector in upstream direction
            (boundary_zone isa InFlow) && (span_angle = π)
        elseif isapprox(dot(normalize(spanning_set[:, 1]), flow_direction_), -1.0)
            # Normal vector points in upstream direction.
            # Flip the outflow vector in downstream direction
            (boundary_zone isa OutFlow) && (span_angle = π)
        else
            throw(ArgumentError("Flow direction and normal vector of " *
                                "$(typeof(boundary_zone))-plane do not correspond."))
        end

        spanning_set[:, 1] = rot_matrix(span_angle, Val(NDIMS)) * spanning_set[:, 1]

        spanning_set_ = SMatrix{NDIMS, NDIMS}(spanning_set)

        return new{typeof(boundary_zone), NDIMS, ELTYPE, typeof(viscosity), typeof(buffer),
                   typeof(velocity_function)}(initial_condition, mass, volume, density,
                                              pressure, characteristics,
                                              previous_characteristics, sound_speed,
                                              boundary_zone, interior_system, zone_origin,
                                              spanning_set_, flow_direction_, viscosity,
                                              buffer, velocity_function,
                                              interior_system.acceleration)
    end
end

spanning_vectors(plane_size::SVector{1}, zone_width) = [-zone_width]

function spanning_vectors(plane_size::SVector{2}, zone_width)
    # Calculate normal vector of plane
    b = normalize([-plane_size[2]; plane_size[1]]) * zone_width

    return hcat(b, plane_size)
end

function spanning_vectors(plane_size::SVector{3}, zone_width)
    a = [plane_size[1]; plane_size[2]; zero(eltype(plane_size))]
    b = [plane_size[1]; zero(eltype(plane_size)); plane_size[3]]

    # Calculate normal vector of plane
    c = normalize(cross(b, a)) * zone_width

    return hcat(c, a, b)
end

timer_name(::OpenBoundarySPHSystem) = "open_boundary"

# TODO
function Base.show(io::IO, system::OpenBoundarySPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "OpenBoundarySPHSystem{", ndims(system), "}(")
    print(io, ", ", system.boundary_zone)
    print(io, ", ", system.acceleration)
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
        summary_line(io, "acceleration", system.acceleration)
        summary_footer(io)
    end
end

@inline function particle_density(v, system::OpenBoundarySPHSystem, particle)
    return system.density[particle]
end

@inline function particle_pressure(v, system::OpenBoundarySPHSystem, particle)
    return system.pressure[particle]
end

@inline set_particle_density(particle, v, system::OpenBoundarySPHSystem, density) = system

@inline set_particle_pressure(particle, v, system, pressure) = system

function set_transport_velocity!(system::OpenBoundarySPHSystem,
                                 particle, particle_old, v, v_old)
    return system
end

@inline function smoothing_kernel(system::OpenBoundarySPHSystem, distance)
    (; smoothing_kernel, smoothing_length) = system.interior_system
    return kernel(smoothing_kernel, distance, smoothing_length)
end

struct InFlow end

struct OutFlow end

@inline function within_boundary_zone(particle_coords, system)
    (; zone_origin, spanning_set) = system
    particle_positon = particle_coords - zone_origin

    for dim in 1:ndims(system)
        span_dim = spanning_set[:, dim]
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

update_open_boundary_eachstep!(system, v_ode, u_ode, semi, t) = system

function update_open_boundary_eachstep!(system::OpenBoundarySPHSystem, v_ode, u_ode,
                                        semi, t)
    u = wrap_u(u_ode, system, semi)
    v = wrap_v(v_ode, system, semi)

    update_quantities!(system, v, u, t)

    check_domain!(system, v, u, v_ode, u_ode, semi)

    update!(system.buffer)
    update!(system.interior_system.buffer)
end

update_transport_velocity!(system::OpenBoundarySPHSystem, v_ode, semi) = system

# J1: Associated with convection and entropy and propagates at flow velocity.
# J2: Propagates downstream to the local flow
# J3: Propagates upstream to the local flow
@inline function evaluate_characteristics!(system, v, u, v_ode, u_ode, semi, t)
    (; interior_system, volume, sound_speed, characteristics, velocity_function,
    previous_characteristics, flow_direction, boundary_zone) = system

    system_interior_nhs = get_neighborhood_search(system, interior_system, semi)

    u_interior = wrap_u(u_ode, interior_system, semi)
    v_interior = wrap_v(v_ode, interior_system, semi)

    system_coords = current_coordinates(u, system)
    interior_coords = current_coordinates(u_interior, interior_system)

    for particle in eachparticle(system)
        previous_characteristics[1, particle] = characteristics[1, particle]
        previous_characteristics[2, particle] = characteristics[2, particle]
        previous_characteristics[3, particle] = characteristics[3, particle]
        previous_characteristics[4, particle] = zero(eltype(characteristics))
    end

    set_zero!(characteristics)
    set_zero!(volume)

    # Loop over all interior neighbors within the kernel cutoff.
    for_particle_neighbor(system, interior_system, system_coords, interior_coords,
                          system_interior_nhs) do particle, neighbor, pos_diff, distance
        rho = particle_density(v_interior, interior_system, neighbor)
        rho_ref = interior_system.initial_condition.density[neighbor]

        p = particle_pressure(v_interior, interior_system, neighbor)
        p_ref = interior_system.initial_condition.pressure[neighbor]

        v_neighbor = current_velocity(v_interior, interior_system, neighbor)

        position = current_coords(u, system, particle)
        # Determine the reference velocity at the position of the interior particle
        v_neighbor_ref = reference_velocity(system, velocity_function, position, t)
        density_term = -sound_speed^2 * (rho - rho_ref)
        pressure_term = p - p_ref
        velocity_term = rho * sound_speed *
                        (dot(v_neighbor - v_neighbor_ref, flow_direction))

        kernel_ = smoothing_kernel(system, distance)

        characteristics[1, particle] += (density_term + pressure_term) * kernel_
        characteristics[2, particle] += (velocity_term + pressure_term) * kernel_
        characteristics[3, particle] += (-velocity_term + pressure_term) * kernel_

        volume[particle] += kernel_

        # Indicate that particle is inside the influence of interior particles.
        previous_characteristics[end, particle] = 1.0
    end

    # Only some of the in-/outlet particles are in the influence of the interior particles.
    # Thus, we find the characteristics for the particle which are outside the influence
    # using the average of the values of the previous time step.
    for particle in each_moving_particle(system)

        # Particle is outside of the influence of interior particles
        if isapprox(previous_characteristics[end, particle], 0.0)

            # Using the average of the values at the previous time step for particles which
            # are outside of the influence of interior particles.
            avg_J1 = 0.0
            avg_J2 = 0.0
            avg_J3 = 0.0
            counter = 0

            for neighbor in each_moving_particle(system)
                # Make sure that only neighbors in the influence of
                # the interior particles are used.
                if isapprox(previous_characteristics[end, neighbor], 1.0)
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

@inline function prescribe_conditions!(characteristics, particle, boundary_zone::OutFlow)
    # J3 is prescribed (i.e. determined from the exterior of the domain).
    # J1 and J2 is transimtted from the domain interior.
    characteristics[3, particle] = zero(eltype(characteristics))

    return characteristics
end

@inline function prescribe_conditions!(characteristics, particle, boundary_zone::InFlow)
    # Allow only J3 to propagate upstream to the boundary
    characteristics[1, particle] = zero(eltype(characteristics))
    characteristics[2, particle] = zero(eltype(characteristics))

    return characteristics
end

@inline function update_quantities!(system::OpenBoundarySPHSystem, v, u, t)
    (; initial_condition, density, pressure, characteristics, flow_direction,
    sound_speed, velocity_function) = system

    for particle in each_moving_particle(system)
        J1 = characteristics[1, particle]
        J2 = characteristics[2, particle]
        J3 = characteristics[3, particle]

        density[particle] = initial_condition.density[particle] +
                            ((-J1 + 0.5 * (J2 + J3)) / sound_speed^2)

        pressure[particle] = initial_condition.pressure[particle] + 0.5 * (J2 + J3)

        particle_position = current_coords(u, system, particle)
        v_ref = reference_velocity(system, velocity_function, particle_position, t)

        particle_velocity = v_ref +
                            ((J2 - J3) / (2 * sound_speed * density[particle])) *
                            flow_direction
        for dim in 1:ndims(system)
            v[dim, particle] = particle_velocity[dim]
        end
    end
end

function check_domain!(system, v, u, v_ode, u_ode, semi)
    (; boundary_zone, interior_system) = system

    neighborhood_search = get_neighborhood_search(system, interior_system, semi)

    u_interior = wrap_u(u_ode, interior_system, semi)
    v_interior = wrap_v(v_ode, interior_system, semi)

    for particle in each_moving_particle(system)
        particle_coords = current_coords(u, system, particle)

        # Check if the particle position is outside the boundary zone.
        if !within_boundary_zone(particle_coords, system)
            transform_particle!(system, interior_system, boundary_zone, particle,
                                v, u, v_interior, u_interior)
        end

        # Check neighbors (only from `interior_system`)
        for interior_neighbor in eachneighbor(particle_coords, neighborhood_search)
            interior_coords = current_coords(u_interior, interior_system, interior_neighbor)

            # Check if particle position is in boundary zone
            if within_boundary_zone(interior_coords, system)
                transform_particle!(interior_system, system, boundary_zone,
                                    interior_neighbor, v, u, v_interior, u_interior)
            end
        end
    end
end

# Outflow particle is outside the boundary zone
@inline function transform_particle!(system::OpenBoundarySPHSystem, interior_system,
                                     ::OutFlow, particle, v, u, v_interior, u_interior)
    deactivate_particle!(system, particle, u)

    return system
end

# Inflow particle is outside the boundary zone
@inline function transform_particle!(system::OpenBoundarySPHSystem, interior_system,
                                     ::InFlow, particle, v, u, v_interior, u_interior)
    (; spanning_set, zone_origin) = system

    # Activate a new particle in simulation domain
    activate_particle!(interior_system, system, particle, v_interior, u_interior, v, u)

    # Reset position and velocity of particle
    for dim in 1:ndims(system)
        u[dim, particle] -= (zone_origin - spanning_set[:, 1])[dim]
    end

    return system
end

# Interior particle is in boundary zone
@inline function transform_particle!(interior_system, system::OpenBoundarySPHSystem,
                                     boundary_zone, particle, v, u, v_interior, u_interior)
    # Activate particle in boundary zone
    activate_particle!(system, interior_system, particle, v, u, v_interior, u_interior)

    # Deactivate particle in interior domain
    deactivate_particle!(interior_system, particle, u_interior)

    return interior_system
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

    # Only when using TVF
    set_transport_velocity!(system_new, particle_new, particle_old, v_new, v_old)

    return system_new
end

@inline function available_particle(system)
    (; active_particle) = system.buffer

    for particle in eachindex(active_particle)
        if !active_particle[particle]
            active_particle[particle] = true

            return particle
        end
    end

    error("No buffer IDs available") # TODO
end

@inline function deactivate_particle!(system, particle, u)
    (; active_particle) = system.buffer

    active_particle[particle] = false
    for dim in 1:ndims(system)
        # Inf or NaN causes instability outcome.
        u[dim, particle] = inv(eps())
    end

    return system
end

function write_v0!(v0, system::OpenBoundarySPHSystem)
    for particle in eachparticle(system)
        # Write particle velocities
        for dim in 1:ndims(system)
            v0[dim, particle] = system.initial_condition.velocity[dim, particle]
        end
    end

    return v0
end

function reference_velocity(system::OpenBoundarySPHSystem, velocity_function, position, t)
    return SVector{ndims(system)}(velocity_function(position, t))
end

function reference_velocity(system::OpenBoundarySPHSystem, ::Nothing, position, t)
    # For a constant velocity field, use the velocity of the first particle
    return extract_svector(system.initial_condition.velocity, system, 1)
end

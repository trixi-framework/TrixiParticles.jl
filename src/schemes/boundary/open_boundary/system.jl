struct OpenBoundarySPHSystem{BZ, NDIMS, ELTYPE <: Real, V, B, VF} <: FluidSystem{NDIMS}
    initial_condition         :: InitialCondition{ELTYPE}
    mass                      :: Array{ELTYPE, 1} # [particle]
    volume                    :: Array{ELTYPE, 1} # [particle]
    density                   :: Array{ELTYPE, 1} # [particle]
    pressure                  :: Array{ELTYPE, 1} # [particle]
    characteristics           :: Array{ELTYPE, 2} # [characteristics, particle]
    previous_characteristics  :: Array{ELTYPE, 2} # [characteristics, particle]
    sound_speed               :: ELTYPE
    boundary_zone             :: BZ
    interior_system           :: System
    zone_origin               :: SVector{NDIMS, ELTYPE}
    zone                      :: Array{ELTYPE, 2}
    unit_normal               :: SVector{NDIMS, ELTYPE}
    viscosity                 :: V
    buffer                    :: B
    initial_velocity_function :: VF
    acceleration              :: SVector{NDIMS, ELTYPE}

    function OpenBoundarySPHSystem(initial_condition, boundary_zone, sound_speed,
                                   zone_plane, zone_origin, interior_system;
                                   initial_velocity_function=nothing,
                                   buffer=nothing)
        (buffer ≠ nothing) && (buffer = SystemBuffer(nparticles(initial_condition), buffer))
        initial_condition = allocate_buffer(initial_condition, buffer)

        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)

        mass = copy(initial_condition.mass)
        pressure = copy(initial_condition.pressure)
        density = copy(initial_condition.density)
        volume = similar(initial_condition.density)

        viscosity = interior_system.viscosity

        characteristics = zeros(ELTYPE, 3, length(mass))
        previous_characteristics = zeros(ELTYPE, 4, length(mass))

        zone_origin_ = SVector{NDIMS}(zone_origin)

        # spans vectors in each direction
        zone = zeros(NDIMS, NDIMS)

        # TODO either check if the vectors are perpendicular to the faces, or obtain perpendicular
        # vectors by using the cross-product (?):
        # zone[1, :] = cross(zone[2, :], zone[3, :])
        # zone[2, :] = cross(zone[1, :], zone[3, :])
        # zone[3, :] = cross(zone[1, :], zone[2, :])
        for dim in 1:NDIMS
            zone[:, dim] .= zone_plane[dim] - zone_origin_
        end

        unit_normal_ = SVector{NDIMS}(normalize(zone[:, 1]))

        return new{typeof(boundary_zone), NDIMS, ELTYPE, typeof(viscosity), typeof(buffer),
                   typeof(initial_velocity_function)}(initial_condition, mass, volume,
                                                      density, pressure, characteristics,
                                                      previous_characteristics, sound_speed,
                                                      boundary_zone, interior_system,
                                                      zone_origin_, zone, unit_normal_,
                                                      viscosity, buffer,
                                                      initial_velocity_function,
                                                      interior_system.acceleration)
    end
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

@inline function set_particle_density(particle, v, system::OpenBoundarySPHSystem, density)
    return system.density[particle] = density
end

@inline function set_particle_pressure(particle, v, system::FluidSystem, pressure)
    return system.pressure[particle] = pressure
end

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

function (boundary_zone::Union{InFlow, OutFlow})(particle_coords, particle, zone_origin,
                                                 zone, system)
    position = particle_coords - zone_origin

    for dim in 1:ndims(system)
        direction = extract_svector(zone, system, dim)

        if !(0 <= dot(position, direction) <= dot(direction, direction))

            # Particle is not in boundary zone.
            return false
        end
    end

    # Particle is in boundary zone.
    return true
end

function update_final!(system::OpenBoundarySPHSystem, v, u, v_ode, u_ode, semi, t)
    evaluate_characteristics!(system, v, u, v_ode, u_ode, semi)
end

update_open_boundary_eachstep!(system, v_ode, u_ode, semi) = system

function update_open_boundary_eachstep!(system::OpenBoundarySPHSystem, v_ode, u_ode, semi)
    u = wrap_u(u_ode, system, semi)
    v = wrap_v(v_ode, system, semi)

    compute_quantities!(system, v)

    check_domain!(system, v, u, v_ode, u_ode, semi)

    update!(system.buffer)
    update!(system.interior_system.buffer)
end

update_transport_velocity!(system::OpenBoundarySPHSystem, v_ode, semi) = system

# J1: Associated with convection and entropy and propagates at flow velocity.
# J2: Propagates downstream to the local flow
# J3: Propagates upstream to the local flow
@inline function evaluate_characteristics!(system, v, u, v_ode, u_ode, semi)
    (; interior_system, volume, sound_speed, characteristics,
    previous_characteristics, unit_normal, boundary_zone) = system

    system_interior_nhs = neighborhood_searches(system, interior_system, semi)
    system_nhs = neighborhood_searches(system, system, semi)

    u_interior = wrap_u(u_ode, interior_system, semi)
    v_interior = wrap_v(v_ode, interior_system, semi)

    system_coords = current_coordinates(u, system)
    interior_coords = current_coordinates(u_interior, interior_system)

    set_zero!(previous_characteristics)

    for particle in eachparticle(system)
        previous_characteristics[1, particle] = characteristics[1, particle]
        previous_characteristics[2, particle] = characteristics[2, particle]
        previous_characteristics[3, particle] = characteristics[3, particle]
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
        v_neighbor_ref = initial_velocity(interior_system, neighbor)

        density_term = -sound_speed^2 * (rho - rho_ref)
        pressure_term = p - p_ref
        velocity_term = rho * sound_speed * (dot(v_neighbor - v_neighbor_ref, unit_normal))

        kernel_ = smoothing_kernel(system, distance)

        evaluate_characteristics_per_particle!(characteristics, particle, density_term,
                                               pressure_term, velocity_term, kernel_,
                                               boundary_zone)
        volume[particle] += kernel_

        # Indicate that particle is inside the influence of interior particles.
        previous_characteristics[end, particle] = 1.0
    end

    (; search_radius) = system_nhs
    for particle in each_moving_particle(system)

        # Particle is outside of the influence of interior particles
        if isapprox(previous_characteristics[end, particle], 0.0)

            # Using the average of the values at the previous time step for particles which
            # are outside of the influence of interior particles.
            particle_coords = current_coords(u, system, particle)
            avg_J1 = 0.0
            avg_J2 = 0.0
            avg_J3 = 0.0
            counter = 0

            for neighbor in eachneighbor(particle_coords, system_nhs)
                neighbor_coords = current_coords(u, system, neighbor)
                pos_diff = particle_coords - neighbor_coords
                distance2 = dot(pos_diff, pos_diff)

                if distance2 <= search_radius^2 &&
                   isapprox(previous_characteristics[end, neighbor], 1.0)
                    avg_J1 += previous_characteristics[1, neighbor]
                    avg_J2 += previous_characteristics[2, neighbor]
                    avg_J3 += previous_characteristics[3, neighbor]
                    counter += 1
                end
            end

            if counter > 0
                characteristics[1, particle] = avg_J1 / counter
                characteristics[2, particle] = avg_J2 / counter
                characteristics[3, particle] = avg_J3 / counter
            end
        else
            characteristics[1, particle] /= volume[particle]
            characteristics[2, particle] /= volume[particle]
            characteristics[3, particle] /= volume[particle]
        end
    end

    return system
end

@inline function evaluate_characteristics_per_particle!(characteristics, particle,
                                                        density_term, pressure_term,
                                                        velocity_term, kernel_weight,
                                                        boundary_zone::OutFlow)
    # J3 is prescribed (i.e. determined from the exterior of the domain).
    # J1 and J2 is transimtted from the domain interior.
    characteristics[1, particle] += (density_term + pressure_term) * kernel_weight
    characteristics[2, particle] += (velocity_term + pressure_term) * kernel_weight

    return characteristics
end

@inline function evaluate_characteristics_per_particle!(characteristics, particle,
                                                        density_term, pressure_term,
                                                        velocity_term, kernel_weight,
                                                        boundary_zone::InFlow)
    # Allow only J3 to propagate upstream to the boundary
    characteristics[3, particle] += (-velocity_term + pressure_term) * kernel_weight

    return characteristics
end

@inline function compute_quantities!(system, v)
    (; initial_condition, density, pressure, characteristics, unit_normal,
    sound_speed) = system

    for particle in each_moving_particle(system)
        J1 = characteristics[1, particle]
        J2 = characteristics[2, particle]
        J3 = characteristics[3, particle]

        density[particle] = initial_condition.density[particle] +
                            ((-J1 + 0.5 * (J2 + J3)) / sound_speed^2)
        pressure[particle] = initial_condition.pressure[particle] + 0.5 * (J2 + J3)

        particle_velocity = initial_velocity(system, particle) +
                            ((J2 - J3) /
                             (2 * sound_speed * density[particle])) * unit_normal
        for dim in 1:ndims(system)
            v[dim, particle] = particle_velocity[dim]
        end
    end
end

function check_domain!(system, v, u, v_ode, u_ode, semi)
    (; boundary_zone, zone, zone_origin, interior_system) = system

    neighborhood_search = neighborhood_searches(system, interior_system, semi)

    u_interior = wrap_u(u_ode, interior_system, semi)
    v_interior = wrap_v(v_ode, interior_system, semi)

    for particle in each_moving_particle(system)
        particle_coords = current_coords(u, system, particle)

        # Check if the particle position is outside the boundary zone.
        if !boundary_zone(particle_coords, particle, zone_origin, zone, system)
            transform_particle!(system, interior_system, boundary_zone, particle,
                                v, u, v_interior, u_interior)
        end

        # Check neighbors (only from `interior_system`)
        for interior_neighbor in eachneighbor(particle_coords, neighborhood_search)
            interior_coords = current_coords(u_interior, interior_system, interior_neighbor)

            # Check if particle position is in boundary zone
            if boundary_zone(interior_coords, interior_neighbor, zone_origin, zone,
                             interior_system)
                transform_particle!(interior_system, system, boundary_zone,
                                    interior_neighbor, v, u, v_interior, u_interior)
            end
        end
    end
end

# Outflow particle is outside the boundary zone
@inline function transform_particle!(system::OpenBoundarySPHSystem, interior_system,
                                     ::OutFlow, particle,
                                     v, u, v_interior, u_interior)
    deactivate_particle!(system, particle, u)

    return system
end

# Inflow particle is outside the boundary zone
@inline function transform_particle!(system::OpenBoundarySPHSystem, interior_system,
                                     ::InFlow, particle,
                                     v, u, v_interior, u_interior)
    (; zone) = system

    # Activate a new particle in simulation domain
    activate_particle!(interior_system, system, particle, v_interior, u_interior, v, u)

    # Reset position and velocity of particle
    u_particle_ref = current_coords(u, system, particle) -
                     extract_svector(zone, system, 1)
    v_particle_ref = initial_velocity(system, particle)

    for dim in 1:ndims(system)
        u[dim, particle] = u_particle_ref[dim]
        v[dim, particle] = v_particle_ref[dim]
    end

    return system
end

# Interior particle is in boundary zone
@inline function transform_particle!(interior_system, system::OpenBoundarySPHSystem,
                                     boundary_zone, particle,
                                     v, u, v_interior, u_interior)

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
    density_ref = system_old.initial_condition.density[particle_old]
    system_new.initial_condition.density[particle_new] = density_ref

    # Exchange pressure
    pressure = particle_pressure(v_old, system_old, particle_old)
    set_particle_pressure(particle_new, v_new, system_new, pressure)
    pressure_ref = system_old.initial_condition.pressure[particle_old]
    system_new.initial_condition.pressure[particle_new] = pressure_ref

    v_ref_new = initial_velocity(system_old, particle_old)

    # Exchange position and velocity
    for dim in 1:ndims(system_new)
        u_new[dim, particle_new] = u_old[dim, particle_old]
        v_new[dim, particle_new] = v_old[dim, particle_old]
        system_new.initial_condition.velocity[dim, particle_new] = v_ref_new[dim]
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
        v_init = initial_velocity(system, particle)
        # Write particle velocities
        for dim in 1:ndims(system)
            v0[dim, particle] = v_init[dim]
        end
    end

    return v0
end

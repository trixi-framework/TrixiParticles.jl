struct OpenBoundarySPHSystem{BZ, NDIMS, ELTYPE <: Real, V} <: FluidSystem{NDIMS}
    initial_condition        :: InitialCondition{ELTYPE}
    mass                     :: Array{ELTYPE, 1} # [particle]
    volume                   :: Array{ELTYPE, 1} # [particle]
    density                  :: Array{ELTYPE, 1} # [particle]
    pressure                 :: Array{ELTYPE, 1} # [particle]
    characteristics          :: Array{ELTYPE, 2} # [characteristics, particle]
    previous_characteristics :: Array{ELTYPE, 2} # [characteristics, particle]
    sound_speed              :: ELTYPE
    boundary_zone            :: BZ
    in_domain                :: BitVector
    interior_system          :: System
    zone_origin              :: SVector{NDIMS, ELTYPE}
    zone                     :: Array{ELTYPE, 2}
    unit_normal              :: SVector{NDIMS, ELTYPE}
    viscosity                :: V
    acceleration             :: SVector{NDIMS, ELTYPE}

    function OpenBoundarySPHSystem(initial_condition, boundary_zone, sound_speed,
                                   zone_points, zone_origin, interior_system,
                                   viscosity=NoViscosity())
        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)

        mass = copy(initial_condition.mass)
        pressure = copy(initial_condition.pressure)
        density = copy(initial_condition.density)
        volume = similar(initial_condition.density)

        characteristics = zeros(ELTYPE, 3, length(mass))
        previous_characteristics = similar(characteristics)
        in_domain = trues(length(mass))

        zone_origin_ = SVector{NDIMS}(zone_origin)
        zone = zeros(NDIMS, NDIMS)

        # TODO either check if the vectors are perpendicular to the faces, or obtain perpendicular
        # vectors by using the cross-product (?):
        # zone[1, :] = cross(zone[2, :], zone[3, :])
        # zone[2, :] = cross(zone[1, :], zone[3, :])
        # zone[3, :] = cross(zone[1, :], zone[2, :])
        for dim in 1:NDIMS
            zone[:, dim] .= zone_points[dim] - zone_origin_
        end

        unit_normal_ = SVector{NDIMS}(normalize(zone[:, 1]))

        return new{typeof(boundary_zone), NDIMS, ELTYPE,
                   typeof(viscosity)}(initial_condition, mass, volume, density, pressure,
                                      characteristics, previous_characteristics,
                                      sound_speed, boundary_zone, in_domain,
                                      interior_system, zone_origin_, zone, unit_normal_,
                                      viscosity, interior_system.acceleration)
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

@inline function set_particle_density(particle, v, system::OpenBoundarySPHSystem, density)
    return system.density[particle] = density
end

@inline function set_particle_pressure(particle, v, system::FluidSystem, pressure)
    return system.pressure[particle] = pressure
end

@inline function smoothing_kernel(system::OpenBoundarySPHSystem, distance)
    @unpack smoothing_kernel, smoothing_length = system.interior_system
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

function (boundary_zone::OutFlow)(particle_coords, particle, zone_origin, zone,
                                  system::OpenBoundarySPHSystem)
    position = particle_coords - zone_origin
    for dim in 1:ndims(system)
        direction = extract_svector(zone, system, dim)

        if !(0 <= dot(position, direction) <= dot(direction, direction))

            # particle is out of domain
            system.in_domain[particle] = false

            # Particle is not in boundary zone.
            return false
        end
    end

    # Particle is in boundary zone.
    return true
end

function update_open_boundary!(system::OpenBoundarySPHSystem, system_index, v_ode, u_ode,
                               semi)
    u = wrap_u(u_ode, system_index, system, semi)
    v = wrap_v(v_ode, system_index, system, semi)

    evaluate_characteristics!(system, system_index, u, u_ode, v_ode, semi)

    compute_quantities!(system, v)

    check_domain!(system, system_index, v, u, v_ode, u_ode, semi)

    update!(system.initial_condition.buffer)
    update!(system.interior_system.initial_condition.buffer)
end

@inline function evaluate_characteristics!(system, system_index, u, u_ode, v_ode, semi)
    @unpack interior_system, volume, initial_condition, sound_speed, characteristics,
    previous_characteristics, unit_normal, boundary_zone = system
    @unpack neighborhood_searches = semi

    interior_index = semi.system_indices[interior_system]
    system_nhs = neighborhood_searches[system_index][interior_index]
    interior_nhs = neighborhood_searches[interior_index][system_index]

    u_interior = wrap_u(u_ode, interior_index, interior_system, semi)
    v_interior = wrap_v(v_ode, interior_index, interior_system, semi)

    system_coords = current_coordinates(u, system)
    interior_coords = current_coordinates(u_interior, interior_system)

    previous_characteristics .= characteristics
    set_zero!(characteristics)
    set_zero!(volume)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    for_particle_neighbor(system, interior_system, system_coords, interior_coords,
                          system_nhs) do particle, neighbor, pos_diff, distance
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
    end

    for particle in each_moving_particle(system)
        if volume[particle] < sqrt(eps())

            # Using the average of the values at the previous time step for particles which
            # are outside of the influence of the fluid particles.
            particle_coords = current_coords(u, system, particle)
            avg_J1 = 0.0
            avg_J2 = 0.0
            avg_J3 = 0.0
            counter = 0

            for neighbor in eachneighbor(particle_coords, interior_nhs)
                avg_J1 += previous_characteristics[1, neighbor]
                avg_J2 += previous_characteristics[2, neighbor]
                avg_J3 += previous_characteristics[3, neighbor]
                counter += 1
            end
            characteristics[1, particle] = avg_J1 / counter
            characteristics[2, particle] = avg_J2 / counter
            characteristics[3, particle] = avg_J3 / counter

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
                                                        velocity_term, kernel_,
                                                        boundary_zone::OutFlow)
    characteristics[1, particle] += (density_term + pressure_term) * kernel_
    characteristics[2, particle] += (velocity_term + pressure_term) * kernel_

    return characteristics
end

@inline function evaluate_characteristics_per_particle!(characteristics, particle,
                                                        density_term, pressure_term,
                                                        velocity_term, kernel_,
                                                        boundary_zone::InFlow)
    characteristics[3, particle] += (-velocity_term + pressure_term) * kernel_

    return characteristics
end

@inline function compute_quantities!(system, v)
    @unpack initial_condition, density, pressure, characteristics, unit_normal,
    interior_system, sound_speed = system

    for particle in each_moving_particle(system)
        density[particle] = initial_condition.density[particle] +
                            ((-characteristics[1, particle] +
                              0.5 *
                              (characteristics[2, particle] + characteristics[3, particle])) /
                             sound_speed^2)
        pressure[particle] = initial_condition.pressure[particle] +
                             0.5 *
                             (characteristics[2, particle] + characteristics[3, particle])

        particle_velocity = initial_velocity(system, particle) +
                            ((characteristics[2, particle] - characteristics[3, particle]) /
                             (2 * sound_speed * density[particle])) * unit_normal
        for dim in 1:ndims(system)
            v[dim, particle] = particle_velocity[dim]
        end
    end
end

function check_domain!(system, system_index, v, u, v_ode, u_ode, semi)
    @unpack boundary_zone, zone, zone_origin, interior_system = system
    @unpack neighborhood_searches = semi

    interior_index = semi.system_indices[interior_system]
    neighborhood_search = neighborhood_searches[system_index][interior_index]

    u_interior = wrap_u(u_ode, interior_index, interior_system, semi)
    v_interior = wrap_v(v_ode, interior_index, interior_system, semi)

    for particle in each_moving_particle(system)
        particle_coords = current_coords(u, system, particle)

        # check if particle position is out of boundary zone
        if !boundary_zone(particle_coords, particle, zone_origin, zone, system)
            transform_particle!(system, interior_system, particle,
                                v, u, v_interior, u_interior)
        end

        # check neighbors (only from `interior_system`)
        for interior_neighbor in eachneighbor(particle_coords, neighborhood_search)
            interior_coords = current_coords(u_interior, interior_system, interior_neighbor)

            # check if particle position is in boundary zone
            if boundary_zone(interior_coords, interior_neighbor, zone_origin, zone,
                             interior_system)
                transform_particle!(interior_system, system, interior_neighbor,
                                    v, u, v_interior, u_interior)
            end
        end
    end
end

# particle is out of boundary zone
@inline function transform_particle!(system::OpenBoundarySPHSystem,
                                     interior_system, particle,
                                     v, u, v_interior, u_interior)
    @unpack in_domain, zone = system

    if in_domain[particle]

        # activate a new particle in simulation domain
        activate_particle!(interior_system, system, particle, v_interior, u_interior, v, u)

        # reset position and velocity of particle
        u_particle_ref = current_coords(u, system, particle) -
                         extract_svector(zone, system, 1)
        v_particle_ref = initial_velocity(system, particle)

        for dim in 1:ndims(system)
            u[dim, particle] = u_particle_ref[dim]
            v[dim, particle] = v_particle_ref[dim]
        end

        return system
    end

    deactivate_particle!(system, particle, u)

    return system
end

# interior particle is in boundary zone
@inline function transform_particle!(interior_system, system, particle,
                                     v, u, v_interior, u_interior)

    # activate particle in boundary zone
    activate_particle!(system, interior_system, particle, v, u, v_interior, u_interior)

    # deactivate particle in interior domain
    deactivate_particle!(interior_system, particle, u_interior)

    return interior_system
end

@inline function activate_particle!(system_new, system_old, particle_old,
                                    v_new, u_new, v_old, u_old)
    particle_new = available_particle(system_new)

    # exchange densities
    density = particle_density(v_old, system_old, particle_old)
    set_particle_density(particle_new, v_new, system_new, density)
    density_ref = system_old.initial_condition.density[particle_old]
    system_new.initial_condition.density[particle_new] = density_ref

    # exchange pressure
    pressure = particle_pressure(v_old, system_old, particle_old)
    set_particle_pressure(particle_new, v_new, system_new, pressure)
    pressure_ref = system_old.initial_condition.pressure[particle_old]
    system_new.initial_condition.pressure[particle_new] = pressure_ref

    v_ref_new = initial_velocity(system_old, particle_old)

    # exchange position and velocity
    for dim in 1:ndims(system_new)
        u_new[dim, particle_new] = u_old[dim, particle_old]
        v_new[dim, particle_new] = v_old[dim, particle_old]
        system_new.initial_condition.velocity[dim, particle_new] = v_ref_new[dim]
    end

    return system_new
end

@inline function available_particle(system)
    @unpack buffer = system.initial_condition
    @unpack active_particle = buffer

    for particle in eachindex(active_particle)
        if !active_particle[particle]
            active_particle[particle] = true

            return particle
        end
    end

    error("No buffer IDs available") # TODO
end

@inline function deactivate_particle!(system, particle, u)
    @unpack buffer = system.initial_condition

    buffer.active_particle[particle] = false
    for dim in 1:ndims(system)
        # TODO
        # typemax(Int) causes problems with visualisation. For testing use big number
        u[dim, particle] = -(100.0 + rand(1:0.5:2, 1)[1])
    end

    return system
end

function write_v0!(v0, system::OpenBoundarySPHSystem)
    @unpack initial_condition = system

    for particle in eachparticle(system)
        # Write particle velocities
        for dim in 1:ndims(system)
            v0[dim, particle] = initial_condition.velocity[dim, particle]
        end
    end

    return v0
end

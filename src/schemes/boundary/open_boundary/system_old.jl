struct OpenBoundarySPHSystem{NDIMS, ELTYPE <: Real, B, PB, C} <: System{NDIMS}
    initial_coordinates      :: Array{ELTYPE, 2} # [dimension, particle]
    initial_velocity         :: Array{ELTYPE, 2} # [dimension, particle]
    mass                     :: Array{ELTYPE, 1} # [particle]
    pressure                 :: Array{ELTYPE, 1} # [particle]
    acceleration             :: SVector{NDIMS, ELTYPE}
    J                        :: Array{ELTYPE, 2} # [J, particle]
    previous_characteristics :: Array{ELTYPE, 2} # [J, particle]
    prescribed_pressure      :: ELTYPE
    prescribed_density       :: ELTYPE
    prescribed_velocity      :: ELTYPE
    buffer                   :: B
    boundary_zone            :: PB
    in_domain                :: BitVector
    interior_system          :: System
    cache                    :: C

    # convenience constructor for passing a setup as first argument
    function OpenBoundarySPHSystem(setup, prescribed_pressure, prescribed_density,
                                   prescribed_velocity, boundary_zone, buffer,
                                   interior_system;
                                   acceleration=ntuple(_ -> 0.0, ndims(setup)))
        return OpenBoundarySPHSystem(setup.coordinates, setup.velocities,
                                     setup.masses, setup.densities,
                                     prescribed_pressure, prescribed_density,
                                     prescribed_velocity,
                                     boundary_zone, buffer, interior_system;
                                     acceleration=acceleration)
    end
    function OpenBoundarySPHSystem(coordinates, velocities, masses, densities,
                                   prescribed_pressure, prescribed_density,
                                   prescribed_velocity,
                                   boundary_zone, buffer, interior_system;
                                   acceleration=ntuple(_ -> 0.0,
                                                       size(coordinates,
                                                            1)))
        NDIMS = size(coordinates, 1)
        ELTYPE = eltype(masses)

        coordinates,
        velocities,
        masses,
        density,
        pressures = allocate_buffer(buffer, coordinates, velocities, masses, densities,
                                    prescribed_pressure * ones(size(densities)))

        J = zeros(ELTYPE, 3, length(masses))
        previous_characteristics = Array{ELTYPE, 2}(undef, 3, length(masses))

        in_domain = trues(length(masses))

        cache = (; density)

        # Make acceleration an SVector
        acceleration_ = SVector(acceleration...)
        if length(acceleration_) != NDIMS
            error("Acceleration must be of length $NDIMS for a $(NDIMS)D problem")
        end

        return new{NDIMS, ELTYPE, typeof(buffer),
                   typeof(boundary_zone), typeof(cache)}(coordinates,
                                                         velocities,
                                                         masses,
                                                         pressures,
                                                         acceleration,
                                                         J,
                                                         previous_characteristics,
                                                         prescribed_pressure,
                                                         prescribed_density,
                                                         prescribed_velocity,
                                                         buffer,
                                                         boundary_zone,
                                                         in_domain,
                                                         interior_system,
                                                         cache)
    end
end

struct InFlow{NDIMS, ELTYPE <: Real}
    # TODO: boundary plane instead of ranges
    boundary_range :: NTuple{2, ELTYPE}
    start_postion  :: ELTYPE
    dim            :: Int # not necessary anymore when using boundary_plane and boundary_normal

    function InFlow(boundary_range; dim=1, start_postion=boundary_range[1])
        NDIMS = size(start_postion, 1)

        return new{NDIMS, eltype(boundary_range)}(boundary_range, start_postion, dim)
    end
end

struct OutFlow{NDIMS, ELTYPE <: Real}
    # TODO: boundary plane instead of ranges
    boundary_range :: NTuple{2, ELTYPE}
    start_postion  :: ELTYPE
    dim            :: Int # not necessary anymore when using boundary_plane and boundary_normal

    function OutFlow(boundary_range; dim=1, start_postion=boundary_range[1])
        NDIMS = size(start_postion, 1)

        return new{NDIMS, eltype(boundary_range)}(boundary_range, start_postion, dim)
    end
end

function (boundary_zone::Union{InFlow, OutFlow})(particle_coords, particle, system)
    @unpack boundary_range, dim = boundary_zone

    if (boundary_range[1] < particle_coords[dim] < boundary_range[2])
        # particle is in boundary zone
        return true
    end

    return false
end

function (boundary_zone::InFlow)(particle_coords, particle,
                                 system::OpenBoundarySPHSystem)
    @unpack boundary_range, dim = boundary_zone

    # particle is not in boundary zone
    if !(boundary_range[1] <= particle_coords[dim] < boundary_range[2])
        return true
    end

    return false
end

function (boundary_zone::OutFlow)(particle_coords, particle,
                                  system::OpenBoundarySPHSystem)
    @unpack boundary_range, dim = boundary_zone

    # particle is not in boundary zone
    if !(boundary_range[1] <= particle_coords[dim] < boundary_range[2])

        # particle is out of domain
        system.in_domain[particle] = false

        return true
    end

    return false
end

function update!(system::OpenBoundarySPHSystem, system_index, v, u, v_ode, u_ode, semi, t)
    @unpack boundary_zone = system

    evaluate_characteristics!(boundary_zone, system, system_index, u, u_ode, v_ode,
                              semi)

    compute_quantities!(system, v)

    check_domain(system, system_index, v, u, v_ode, u_ode, semi)

    update!(system.buffer)
    update!(system.interior_system.buffer)

    return system
end

@inline function compute_quantities!(system, v)
    @unpack prescribed_density, prescribed_pressure, prescribed_velocity,
    cache, pressure, J, interior_system, boundary_zone = system
    @unpack density = cache
    @unpack state_equation = interior_system
    @unpack sound_speed = state_equation
    @unpack dim = boundary_zone

    sound_speed_sqrd = sound_speed^2

    for particle in each_moving_particle(system)
        density[particle] = prescribed_density +
                            (-J[1, particle] + 0.5 * (J[2, particle] + J[3, particle])) /
                            sound_speed_sqrd
        v[dim, particle] = prescribed_velocity +
                           (J[2, particle] - J[3, particle]) /
                           (2 * sound_speed * density[particle])
        pressure[particle] = prescribed_pressure + 0.5 * (J[2, particle] + J[3, particle])
    end
end

@inline function evaluate_characteristics!(boundary_zone::InFlow, system,
                                           system_index, u, u_ode, v_ode,
                                           semi)
    @unpack J, interior_system, previous_characteristics,
    prescribed_pressure, prescribed_velocity = system
    @unpack state_equation, smoothing_kernel, smoothing_length,
    pressure = interior_system
    @unpack sound_speed = state_equation
    @unpack dim = boundary_zone
    @unpack neighborhood_searches = semi

    neighborhood_search = neighborhood_searches[system_index][system_index]
    neighbor_index = semi.system_indices[interior_system]
    interior_neighborhood_search = neighborhood_searches[system_index][neighbor_index]

    u_interior = wrap_u(u_ode, neighbor_index, interior_system, semi)
    v_interior = wrap_v(v_ode, neighbor_index, interior_system, semi)

    for particle in each_moving_particle(system)
        particle_coords = get_current_coords(particle, u, system)

        previous_characteristics[3, particle] = J[3, particle]
        J[:, particle] .= 0.0
        volume = 0.0

        for interior_neighbor in eachneighbor(particle_coords, interior_neighborhood_search)
            pos_diff = particle_coords -
                       get_current_coords(interior_neighbor, u_interior, interior_system)
            distance = norm(pos_diff)
            if distance <= compact_support(smoothing_kernel, smoothing_length)
                density = get_particle_density(interior_neighbor, v_interior,
                                               interior_system)
                kernel_ = kernel(smoothing_kernel, distance, smoothing_length)

                pressure_term = (pressure[interior_neighbor] - prescribed_pressure)
                velocity_term = (v_interior[dim, interior_neighbor] - prescribed_velocity)

                # upstream-running characteristics
                J[3, particle] += (-density * sound_speed * velocity_term +
                                   pressure_term) * kernel_
                volume += kernel_
            end
        end

        if volume > sqrt(eps())
            J[3, particle] /= volume
        end
    end

    # using the average of the values at the previous time step for particles which are
    # outside of the influence of the fluid particles.
    for particle in each_moving_particle(system)
        if J[3, particle] < sqrt(eps())
            particle_coords = get_current_coords(particle, u, system)
            avg_J3 = 0.0
            counter = 0
            for neighbor in eachneighbor(particle_coords, neighborhood_search)
                # TODO: Check if it has an influence when only the particles considered which
                # are inside of the influence of the fluid particles (see Negi et al 2020, p.7)
                avg_J3 += previous_characteristics[3, neighbor]
                counter += 1
            end
            J[3, particle] = avg_J3 / counter
        end
    end
    return boundary_zone
end

@inline function evaluate_characteristics!(boundary_zone::OutFlow, system,
                                           system_index, u, u_ode, v_ode,
                                           semi)
    @unpack J, interior_system, previous_characteristics,
    prescribed_pressure, prescribed_density, prescribed_velocity = system
    @unpack state_equation, smoothing_kernel, smoothing_length,
    pressure = interior_system
    @unpack sound_speed = state_equation
    @unpack dim = boundary_zone
    @unpack neighborhood_searches = semi

    neighborhood_search = neighborhood_searches[system_index][system_index]
    neighbor_index = semi.system_indices[interior_system]
    interior_neighborhood_search = neighborhood_searches[system_index][neighbor_index]

    u_interior = wrap_u(u_ode, neighbor_index, interior_system, semi)
    v_interior = wrap_v(v_ode, neighbor_index, interior_system, semi)

    for particle in each_moving_particle(system)
        particle_coords = get_current_coords(particle, u, system)

        previous_characteristics[1, particle] = J[1, particle]
        previous_characteristics[2, particle] = J[2, particle]
        J[:, particle] .= 0.0
        volume = 0.0

        for interior_neighbor in eachneighbor(particle_coords, interior_neighborhood_search)
            pos_diff = particle_coords -
                       get_current_coords(interior_neighbor, u_interior, interior_system)
            distance = norm(pos_diff)
            if distance <= compact_support(smoothing_kernel, smoothing_length)
                density = get_particle_density(interior_neighbor, v_interior,
                                               interior_system)
                kernel_ = kernel(smoothing_kernel, distance, smoothing_length)

                density_term = (density - prescribed_density)
                pressure_term = (pressure[interior_neighbor] - prescribed_pressure)
                velocity_term = (v_interior[dim, interior_neighbor] - prescribed_velocity)

                # downstream-running characteristics
                J[1, particle] += (-sound_speed^2 * density_term + pressure_term) * kernel_
                J[2, particle] += (density * sound_speed * velocity_term + pressure_term) *
                                  kernel_
                volume += kernel_
            end
        end

        if volume > sqrt(eps())
            J[1, particle] /= volume
            J[2, particle] /= volume
        end
    end

    # using the average of the values at the previous time step for particles which are
    # outside of the influence of the fluid particles.
    for particle in each_moving_particle(system)
        if J[1, particle] < sqrt(eps())
            particle_coords = get_current_coords(particle, u, system)
            avg_J1 = 0.0
            avg_J2 = 0.0
            counter = 0
            for neighbor in eachneighbor(particle_coords, neighborhood_search)
                # TODO: Check if it has an influence when only the particles considered which
                # are inside of the influence of the fluid particles (see Negi et al 2020, p.7)
                avg_J1 += previous_characteristics[1, neighbor]
                avg_J2 += previous_characteristics[2, neighbor]
                counter += 1
            end
            J[1, particle] = avg_J1 / counter
            J[2, particle] = avg_J2 / counter
        end
    end
    return boundary_zone
end

@inline function check_domain(system, system_index, v, u, v_ode, u_ode, semi)
    @unpack boundary_zone, interior_system = system
    @unpack particle_systems, neighborhood_searches = semi

    neighbor_index = semi.system_indices[interior_system]
    neighborhood_search = neighborhood_searches[system_index][neighbor_index]

    u_interior = wrap_u(u_ode, neighbor_index, interior_system, semi)
    v_interior = wrap_v(v_ode, neighbor_index, interior_system, semi)

    for particle in each_moving_particle(system)
        particle_coords = get_particle_coords(particle, u, system)

        # check if particle position is out of boundary zone
        if boundary_zone(particle_coords, particle, system)
            transform_particle(system, interior_system, particle,
                               boundary_zone, v, u, v_interior, u_interior)
        end
        # check neighbors (only from interior_system)
        for interior_neighbor in eachneighbor(particle_coords, neighborhood_search)
            neighbor_coords = get_particle_coords(interior_neighbor, u_interior,
                                                  system)
            if boundary_zone(neighbor_coords, interior_neighbor, interior_system)
                transform_particle(interior_system, system, interior_neighbor,
                                   boundary_zone, v, u, v_interior, u_interior)
            end
        end
    end
end

# particle is out of boundary zone
@inline function transform_particle(system::OpenBoundarySPHSystem,
                                    interior_system, particle_ID, boundary_zone, v, u,
                                    v_interior, u_interior)
    @unpack in_domain = system
    @unpack dim, start_postion = boundary_zone

    if in_domain[particle_ID]

        # activate a new particle in simulation domain
        activate_particle(system, interior_system, particle_ID,
                          v, u, v_interior, u_interior)

        # reset position and velocity of particle
        u[dim, particle_ID] = start_postion
        for i in 1:ndims(system)
            v[i, particle_ID] = system.initial_velocity[i, particle_ID]
        end

        return system
    end

    deactivate_particle(particle_ID, system, u)

    return system
end

# interior particle is in boundary zone
@inline function transform_particle(interior_system, system, particle_ID,
                                    boundary_zone, v, u, v_interior, u_interior)

    # activate particle in boundary zone
    activate_particle(interior_system, system, particle_ID,
                      v_interior, u_interior, v, u)

    # deactivate particle in interior domain
    deactivate_particle(particle_ID, interior_system, u_interior)

    return interior_system
end

@inline function activate_particle(system, interior_system,
                                   particle_ID, v, u, v_interior, u_interior)
    new_particle_ID = get_available_ID(interior_system)

    # exchange densities
    set_particle_density(system, interior_system, particle_ID, new_particle_ID,
                         v, v_interior)

    # exchange pressure
    interior_system.pressure[new_particle_ID] = system.pressure[particle_ID]

    # exchange position and velocity
    for dim in 1:ndims(system)
        u_interior[dim, new_particle_ID] = u[dim, particle_ID]
        v_interior[dim, new_particle_ID] = v[dim, particle_ID]
    end

    return system
end

@inline function get_available_ID(system)
    @unpack buffer = system
    @unpack active_particle = buffer

    for particle_ID in eachindex(active_particle)
        if !active_particle[particle_ID]
            active_particle[particle_ID] = true

            return particle_ID
        end
    end

    error("No buffer IDs available") # TODO
end

@inline function set_particle_density(system::OpenBoundarySPHSystem,
                                      interior_system, particle_ID,
                                      new_particle_ID, v, v_interior)
    @unpack density_calculator = interior_system

    set_particle_density(particle_ID, new_particle_ID, v, v_interior, density_calculator,
                         system, interior_system)
end

@inline function set_particle_density(interior_system, system, particle_ID,
                                      new_particle_ID, v, v_interior)
    system.cache.density[new_particle_ID] = get_particle_density(particle_ID,
                                                                 v_interior,
                                                                 interior_system)
end

@inline function set_particle_density(particle_ID, new_particle_ID, v, v_interior,
                                      ::SummationDensity, system, interior_system)
    # density is in cache for each particle
    interior_system.cache.density[new_particle_ID] = system.cache.density[particle_ID]
end

@inline function set_particle_density(particle_ID, new_particle_ID, v, v_interior,
                                      ::ContinuityDensity, system, interior_system)
    v_interior[end, new_particle_ID] = system.cache.density[particle_ID]
end

@inline function deactivate_particle(particle_ID, system, u)
    @unpack buffer = system

    buffer.active_particle[particle_ID] = false
    for dim in 1:ndims(system)
        # TODO
        # typemax(Int) causes problems with visualisation. For testing use big number
        u[dim, particle_ID] = -(100.0 + rand(1:0.5:2, 1)[1])
    end
end

@inline function get_hydrodynamic_mass(particle,
                                       system::Union{FluidParticleContainer,
                                                     OpenBoundarySPHSystem})
    return system.mass[particle]
end

@inline function get_particle_density(particle, v, system::OpenBoundarySPHSystem)
    return system.cache.density[particle]
end

function write_u0!(u0, system::OpenBoundarySPHSystem)
    @unpack initial_coordinates = system

    for particle in eachparticle(system)
        # Write particle coordinates
        for dim in 1:ndims(system)
            u0[dim, particle] = initial_coordinates[dim, particle]
        end
    end

    return u0
end

function write_v0!(v0, system::OpenBoundarySPHSystem)
    @unpack initial_velocity = system

    for particle in eachparticle(system)
        # Write particle velocities
        for dim in 1:ndims(system)
            v0[dim, particle] = initial_velocity[dim, particle]
        end
    end

    return v0
end

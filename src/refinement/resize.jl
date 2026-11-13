# Resizing of systems with particle refinement.
#
# Splitting particles needs more particles than a system was created with. Systems with
# particle refinement and a `SystemBuffer` can therefore grow: the new particles are added
# as inactive buffer particles at the end of the system. Systems can only grow, since
# deleting particles only deactivates them.

"""
    resize_systems!(v_ode, u_ode, semi, n_particles)

Resize the systems of `semi` to `n_particles[i]` particles for the `i`-th system and
resize `v_ode` and `u_ode` accordingly. The new particles are inactive buffer particles,
so `n_particles[i]` is the new buffer capacity of the system.

Only fluid systems with particle refinement and a `SystemBuffer` can be resized, and only
to a larger number of particles. All other systems must keep their number of particles.

The data of all systems in `v_ode` and `u_ode` is kept, and the neighborhood searches
of the resized systems are recreated with the new number of particles.

!!! warning
    This changes the length of `v_ode` and `u_ode`. When called from a callback,
    the integrator has to be resized as well.
"""
function resize_systems!(v_ode, u_ode, semi, n_particles)
    (; systems) = semi

    if length(n_particles) != length(systems)
        throw(ArgumentError("`n_particles` must have one entry per system"))
    end

    resized = [n_particles[i] != nparticles(systems[i]) for i in eachindex(systems)]
    any(resized) || return semi

    # Check all systems before changing anything
    for i in eachindex(systems)
        resized[i] && check_resize(systems[i], n_particles[i], i, semi)
    end

    v_old = copy(v_ode)
    u_old = copy(u_ode)
    ranges_v_old = copy(semi.ranges_v)
    ranges_u_old = copy(semi.ranges_u)
    n_particles_old = [nparticles(system) for system in systems]

    for i in eachindex(systems)
        resized[i] && resize_system!(systems[i], n_particles[i])
    end

    resize_open_boundaries!(semi, resized)

    # The particles of each system are stored in a contiguous range with the particle
    # as the last dimension, so the old data is the beginning of the new range.
    ranges_v, ranges_u = ranges_vu(systems)
    resize!(v_ode, sum(length, ranges_v))
    resize!(u_ode, sum(length, ranges_u))

    for i in eachindex(systems)
        n_v = length(ranges_v_old[i])
        n_u = length(ranges_u_old[i])
        view(v_ode,
             first(ranges_v[i]):(first(ranges_v[i]) + n_v - 1)) .= view(v_old,
                                                                        ranges_v_old[i])
        view(u_ode,
             first(ranges_u[i]):(first(ranges_u[i]) + n_u - 1)) .= view(u_old,
                                                                        ranges_u_old[i])
    end

    copyto!(semi.ranges_v, ranges_v)
    copyto!(semi.ranges_u, ranges_u)

    # Initialize the new particles as inactive particles
    foreach_system(semi) do system
        system_index = system_indices(system, semi)
        resized[system_index] || return

        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)

        for particle in (n_particles_old[system_index] + 1):nparticles(system)
            initialize_inactive_particle!(system, v, u, particle)
        end
    end

    resize_neighborhood_searches!(semi.neighborhood_search_handler, semi, resized)
    initialize_resized_neighborhood_searches!(semi, u_ode, resized)

    return semi
end

function check_resize(system, n, system_index, semi)
    if !(system isa AbstractFluidSystem) || isnothing(get_refinement(system))
        throw(ArgumentError("only fluid systems with particle refinement can be resized, " *
                            "but the system `$(nameof(typeof(system)))` was resized"))
    end

    if !(system.buffer isa SystemBuffer)
        throw(ArgumentError("resizing a system requires a `SystemBuffer`. " *
                            "Pass `buffer_size` to the fluid system."))
    end

    if n < nparticles(system)
        throw(ArgumentError("systems can only grow, but the number of particles of " *
                            "the system `$(nameof(typeof(system)))` would change from " *
                            "$(nparticles(system)) to $n"))
    end

    if !isnothing(system.surface_normal_method) || !isnothing(system.surface_tension)
        throw(ArgumentError("resizing a system with surface normals or surface tension " *
                            "is not supported yet"))
    end

    check_resize_scheme(system)

    return system
end

check_resize_scheme(system) = system

function check_resize_scheme(system::WeaklyCompressibleSPHSystem)
    # TODO: This cache contains matrices, which cannot be resized
    if system.density_diffusion isa DensityDiffusionAntuono
        throw(ArgumentError("resizing a system with `DensityDiffusionAntuono` is not " *
                            "supported yet"))
    end

    return system
end

# Add `n - nparticles(system)` inactive particles to the end of `system`
function resize_system!(system::AbstractFluidSystem, n)
    (; mass, smoothing_length, cache, density_calculator, buffer,
     initial_condition) = system
    refinement = get_refinement(system)
    (; reference_density) = refinement

    # The new particles are initialized like the buffer particles in `allocate_buffer`.
    # Their values are set when they are activated.
    spacing_new = initial_condition.particle_spacing
    mass_new = reference_density * spacing_new^ndims(system)
    resize_and_fill!(mass, n, mass_new)
    resize_and_fill!(smoothing_length, n, initial_smoothing_length(system))

    resize_cache_refinement!(cache, refinement, ndims(system), n, spacing_new)
    resize_cache_density!(cache, density_calculator, n, reference_density)
    resize_cache_shifting!(cache, shifting_technique(system), ndims(system), n)
    resize_scheme_data!(system, n)
    resize_buffer!(buffer, n)

    return system
end

# The open boundary stores data per particle of its fluid system
function resize_open_boundaries!(semi, resized)
    foreach_system(semi) do open_boundary
        open_boundary isa OpenBoundarySystem || return

        fluid_system_index = open_boundary.fluid_system_index[]
        resized[fluid_system_index] || return

        n_fluid = nparticles(semi.systems[fluid_system_index])
        resize_and_fill!(open_boundary.fluid_candidates, n_fluid, false)
    end

    return semi
end

function initialize_inactive_particle!(system, v, u, particle)
    (; reference_density) = get_refinement(system)

    deactivate_particle!(system, particle, v, u)
    set_particle_density!(v, system, particle, reference_density)
    set_particle_pressure!(v, system, particle, zero(eltype(system)))

    return system
end

# Recreate the neighborhood searches with resized neighbor systems for the new number of
# particles. Note that a TLSPH self-interaction is never resized.
function resize_neighborhood_searches!(handler::PairsNHSHandler, semi, resized)
    (; neighborhood_searches) = handler
    (; systems) = semi

    for neighbor_index in eachindex(systems)
        resized[neighbor_index] || continue
        n_points = nparticles(systems[neighbor_index])

        for system_index in axes(neighborhood_searches, 1)
            nhs = neighborhood_searches[system_index, neighbor_index]
            neighborhood_searches[system_index,
                                  neighbor_index] = copy_neighborhood_search(nhs,
                                                                             PointNeighbors.search_radius(nhs),
                                                                             n_points)
        end
    end

    return handler
end

function resize_neighborhood_searches!(handler::SharedNHSHandler, semi, resized)
    (; neighborhood_searches) = handler
    (; systems) = semi

    for neighbor_index in eachindex(systems)
        resized[neighbor_index] || continue
        n_points = nparticles(systems[neighbor_index])
        searches = neighborhood_searches[neighbor_index]

        for radius_index in eachindex(searches)
            nhs = searches[radius_index]
            searches[radius_index] = copy_neighborhood_search(nhs,
                                                              PointNeighbors.search_radius(nhs),
                                                              n_points)
        end
    end

    return handler
end

# Initialize the recreated neighborhood searches with the current coordinates
function initialize_resized_neighborhood_searches!(semi, u_ode, resized)
    foreach_system(semi) do system
        foreach_system(semi) do neighbor
            resized[system_indices(neighbor, semi)] || return

            u_system = wrap_u(u_ode, system, semi)
            u_neighbor = wrap_u(u_ode, neighbor, semi)

            PointNeighbors.initialize!(get_neighborhood_search(system, neighbor, semi),
                                       current_coordinates(u_system, system),
                                       current_coordinates(u_neighbor, neighbor),
                                       eachindex_y=each_active_particle(neighbor),
                                       parallelization_backend=semi.parallelization_backend)
        end
    end

    return semi
end

# Resize `array` to length `n` and fill the new entries with `value`
function resize_and_fill!(array, n, value)
    n_old = length(array)
    resize!(array, n)
    array[(n_old + 1):n] .= value

    return array
end

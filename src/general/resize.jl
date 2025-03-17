# TODO: Make sure that the following functions are only called from a callback
function Base.resize!(semi::Semidiscretization, v_ode, u_ode, _v_ode, _u_ode)
    (; systems) = semi
    # Resize all systems
    foreach_system(semi) do system
        resize!(system, capacity(system))
    end

    copyto!(_v_ode, v_ode)
    copyto!(_u_ode, u_ode)

    # Get ranges after resizing the systems
    ranges_v_new, ranges_u_new = ranges_vu(systems)

    ranges_v_old = copy(semi.ranges_v)
    ranges_u_old = copy(semi.ranges_u)

    # Set ranges after resizing the systems
    for i in 1:length(systems)
        semi.ranges_v[i] = ranges_v_new[i]
        semi.ranges_u[i] = ranges_u_new[i]
    end

    sizes_u = sum(u_nvariables(system) * n_moving_particles(system) for system in systems)
    sizes_v = sum(v_nvariables(system) * n_moving_particles(system) for system in systems)

    # After `deleteat!`, there are fewer particles, and the `deleteat!(system, v, u)` function
    # ensures that all rejected values are stored at the tail of `v_ode` and `u_ode`.
    if length(v_ode) < sizes_v
        # Resize before copy

        resize!(v_ode, sizes_v)
        resize!(_v_ode, sizes_v)

        resize!(u_ode, sizes_u)
        resize!(_u_ode, sizes_u)
    end

    # Copy the values of the old ranges to the new ranges
    for i in eachindex(ranges_u_old)
        length_u = min(length(ranges_u_old[i]), length(ranges_u_new[i]))
        for j in 0:(length_u - 1)
            u_ode[ranges_u_new[i][1] + j] = _u_ode[ranges_u_old[i][1] + j]
        end

        length_v = min(length(ranges_v_old[i]), length(ranges_v_new[i]))
        for j in 0:(length_v - 1)
            v_ode[ranges_v_new[i][1] + j] = _v_ode[ranges_v_old[i][1] + j]
        end
    end

    if length(v_ode) > sizes_v
        # Resize after copy

        resize!(v_ode, sizes_v)
        resize!(u_ode, sizes_u)

        resize!(_v_ode, sizes_v)
        resize!(_u_ode, sizes_u)
    end

    return v_ode
end

Base.resize!(system::System, capacity_system) = system

function Base.resize!(system::FluidSystem, capacity_system)
    error("`resize!`not implemented for $(typeof(system)), yet")
end

function Base.resize!(system::WeaklyCompressibleSPHSystem, capacity_system)
    capacity(system) == nparticles(system) && return system

    if capacity(system) < nparticles(system) && !(system.cache.values_conserved[])
        error("The required capacity is smaller than the actual size of the system. " *
              "Call `deleteat!` before resizing the system, to specify which particles can be deleted")
    end

    (; mass, pressure, density_calculator) = system

    resize!(mass, capacity_system)
    resize!(pressure, capacity_system)

    resize_density!(system, capacity_system, density_calculator)

    resize_cache!(system, capacity_system)

    system.cache.additional_capacity[] = 0
    system.cache.values_conserved[] = false

    return system
end

function Base.resize!(system::EntropicallyDampedSPHSystem, capacity_system)
    capacity(system) == nparticles(system) && return system

    if capacity(system) < nparticles(system) && !(system.cache.values_conserved[])
        error("The required capacity is smaller than the actual size of the system. " *
              "Call `deleteat!` before resizing the system, to specify which particles can be deleted")
    end

    (; mass, density_calculator) = system

    resize!(mass, capacity_system)

    resize_density!(system, capacity_system, density_calculator)

    resize_cache!(system, capacity_system)

    system.cache.additional_capacity[] = 0
    system.cache.values_conserved[] = false

    return system
end

resize_density!(system, n::Int, ::SummationDensity) = resize!(system.cache.density, n)
resize_density!(system, n::Int, ::ContinuityDensity) = system

function resize_cache!(system::WeaklyCompressibleSPHSystem, n::Int)
    # TODO
    # resize!(system.cache.smoothing_length, n)

    return system
end

function resize_cache!(system::EntropicallyDampedSPHSystem, n::Int)
    # TODO
    # resize!(system.cache.smoothing_length, n)
    resize_cache!(system, system.transport_velocity, n)

    return system
end

resize_cache!(system::EntropicallyDampedSPHSystem, ::Nothing, n) = system

function resize_cache!(system::EntropicallyDampedSPHSystem, ::TransportVelocityAdami, n)
    resize!(system.cache.pressure_average, n)
    resize!(system.cache.neighbor_counter, n)
    return system
end

function Base.deleteat!(semi::Semidiscretization, v_ode, u_ode, _v_ode, _u_ode)
    # Delete at specific indices
    foreach_system(semi) do system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)
        deleteat!(system, v, u)
    end

    resize!(semi, v_ode, u_ode, _v_ode, _u_ode)

    return semi
end

Base.deleteat!(system::System, v, u) = system

function Base.deleteat!(system::FluidSystem, v, u)
    (; cache) = system

    isempty(cache.delete_candidates) && return

    delete_counter = 0

    for particle in eachparticle(system)
        if cache.delete_candidates[particle]
            # swap particles (keep -> delete)
            dump_id = nparticles(system) - delete_counter

            vel_keep = current_velocity(v, system, dump_id)
            pos_keep = current_coords(u, system, dump_id)

            mass_keep = hydrodynamic_mass(system, dump_id)
            density_keep = particle_density(v, system, dump_id)
            pressure_keep = particle_pressure(v, system, dump_id)

            # TODO
            # system.cache.smoothing_length[particle] = smoothing_length(system, dump_id)

            system.mass[particle] = mass_keep

            set_particle_pressure!(v, system, particle, pressure_keep)

            set_particle_density!(v, system, particle, density_keep)

            for dim in 1:ndims(system)
                v[dim, particle] = vel_keep[dim]
                u[dim, particle] = pos_keep[dim]
            end

            delete_counter += 1
        end
    end

    cache.additional_capacity[] -= delete_counter
    resize!(cache.delete_candidates, 0)
    cache.values_conserved[] = true

    return system
end

@inline capacity(system) = nparticles(system)

@inline function capacity(system::FluidSystem)
    (; cache) = system
    return nparticles(system) + cache.additional_capacity[]
end

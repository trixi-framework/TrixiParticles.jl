function Base.resize!(semi::Semidiscretization, v_ode, u_ode, _v_ode, _u_ode)
    # Resize all systems
    foreach_system(semi) do system
        capacity(system) > nparticles(system) && resize!(system, capacity(system))
    end

    resize!(v_ode, u_ode, _v_ode, _u_ode, semi)

    return semi
end

function Base.deleteat!(semi::Semidiscretization, v_ode, u_ode, _v_ode, _u_ode)
    # Delete at specific indices
    foreach_system(semi) do system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)
        deleteat!(system, v, u)
    end

    resize!(v_ode, u_ode, _v_ode, _u_ode, semi)

    return semi
end

function Base.resize!(v_ode, u_ode, _v_ode, _u_ode, semi::Semidiscretization)
    (; systems) = semi

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

    sizes_u = sum(u_nvariables(system) * n_moving_particles(system) for system in systems)
    sizes_v = sum(v_nvariables(system) * n_moving_particles(system) for system in systems)

    resize!(v_ode, sizes_v)
    resize!(u_ode, sizes_u)

    resize!(_v_ode, sizes_v)
    resize!(_u_ode, sizes_u)

    # TODO: Do the following in the callback
    # resize!(integrator, (length(v_ode), length(u_ode)))

    # # Tell OrdinaryDiffEq that u has been modified
    # u_modified!(integrator, true)
    return v_ode
end

Base.resize!(system::System, capacity_system) = system

function Base.resize!(system::FluidSystem, capacity_system)
    return resize!(system, system.particle_refinement, capacity_system)
end

Base.resize!(system, ::Nothing, capacity_system) = system

function Base.resize!(system::WeaklyCompressibleSPHSystem, refinement, capacity_system::Int)
    (; mass, pressure, density_calculator) = system

    resize!(mass, capacity_system)
    resize!(pressure, capacity_system)

    resize_density!(system, capacity_system, density_calculator)

    resize_cache!(system, capacity_system)

    resize_refinement!(system)

    return system
end

function Base.resize!(system::EntropicallyDampedSPHSystem, refinement, capacity_system::Int)
    (; mass, density_calculator) = system

    resize!(mass, capacity_system)

    resize_density!(system, capacity_system, density_calculator)

    resize_cache!(system, capacity_system)

    resize_refinement!(system)

    return system
end

resize_density!(system, n::Int, ::SummationDensity) = resize!(system.cache.density, n)
resize_density!(system, n::Int, ::ContinuityDensity) = system

function resize_cache!(system::WeaklyCompressibleSPHSystem, n::Int)
    resize!(system.cache.smoothing_length, n)

    return system
end

function resize_cache!(system::EntropicallyDampedSPHSystem, n::Int)
    resize!(system.cache.smoothing_length, n)
    resize!(system.cache.beta, n)
    resize!(system.cache.pressure_average, n)
    resize!(system.cache.neighbor_counter, n)

    return system
end

Base.deleteat!(system::System, v, u) = system

function Base.deleteat!(system::FluidSystem, v, u)
    return deleteat!(system, system.particle_refinement, v, u)
end

Base.deleteat!(system::FluidSystem, ::Nothing, v, u) = system

function Base.deleteat!(system::FluidSystem, refinement, v, u)
    (; delete_candidates) = refinement

    delete_counter = 0

    for particle in eachparticle(system)
        if !iszero(delete_candidates[particle])
            # swap particles (keep -> delete)
            dump_id = nparticles(system) - delete_counter

            vel_keep = current_velocity(v, system, dump_id)
            pos_keep = current_coords(u, system, dump_id)

            mass_keep = hydrodynamic_mass(system, dump_id)
            density_keep = particle_density(v, system, dump_id)
            pressure_keep = particle_pressure(v, system, dump_id)
            #TODO
            # smoothing_length_keep = smoothing_length(system, dump_id)
            # system.cache.smoothing_length[particle] = smoothing_length_keep

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

    resize!(system, nparticles(system) - delete_counter)

    return system
end

@inline capacity(system) = nparticles(system)

@inline capacity(system::FluidSystem) = capacity(system, system.particle_refinement)

@inline capacity(system, ::Nothing) = nparticles(system)

@inline function capacity(system, particle_refinement)
    return particle_refinement.n_new_particles[] + nparticles(system)
end

function resize_and_copy!(callback, semi, v_ode, u_ode, _v_cache, _u_cache)
    # Get non-`resize!`d ranges
    callback.ranges_v_cache, callback.ranges_u_cache = ranges_vu(semi.systems)
    callback.eachparticle_cache = Tuple(get_iterator(system) for system in semi.systems)
    callback.nparticles_cache = Tuple(n_moving_particles(system) for system in semi.systems)

    # Resize all systems
    foreach_system(semi) do system
        resize_system!(system)
    end

    # Set `resize!`d ranges
    ranges_v_tmp, ranges_u_tmp = ranges_vu(semi.systems)
    for i in 1:length(semi.systems)
        semi.ranges_v[i][1] = ranges_v_tmp[i][1]
        semi.ranges_u[i][1] = ranges_u_tmp[i][1]
    end

    sizes_v = (v_nvariables(system) * n_moving_particles(system) for system in semi.systems)
    sizes_u = (u_nvariables(system) * n_moving_particles(system) for system in semi.systems)

    # Resize integrated values
    resize!(v_ode, sum(sizes_v))
    resize!(u_ode, sum(sizes_u))

    # Preserve non-changing values
    foreach_system(semi) do system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)
        _v = _wrap_v(_v_cache, system, semi, callback)
        _u = _wrap_u(_u_cache, system, semi, callback)

        copy_values_v!(v, _v, system, semi, callback)
        copy_values_u!(u, _u, system, semi, callback)
    end

    return callback
end

@inline resize_system!(system) = system
@inline resize_system!(system::FluidSystem) = resize_system!(system,
                                                             system.particle_refinement)

@inline resize_system!(system::FluidSystem, ::Nothing) = system

@inline function resize_system!(system::FluidSystem,
                                particle_refinement::ParticleRefinement)
    (; system_child, candidates) = particle_refinement

    candidates_refine = length(candidates)

    if candidates_refine != 0 || candidates_coarsen != 0
        n_new_child = candidates_refine * nchilds(system, particle_refinement)
        capacity_parent = nparticles(system) - candidates_refine
        capacity_child = nparticles(system_child) + n_new_child

        capacity_parent < 0 && error("`RefinementCriteria` affects more than all particles")

        # Resize child system (extending)
        resize_system!(system_child, capacity_child)

        # Resize parent system (reducing)
        resize_system!(system, capacity_parent)
    end
end

function resize_system!(system::WeaklyCompressibleSPHSystem, capacity::Int)
    (; mass, pressure, cache, density_calculator) = system

    resize!(mass, capacity)
    resize!(pressure, capacity)
    resize_cache!(cache, capacity, density_calculator)
end

function resize_system!(system::EntropicallyDampedSPHSystem, capacity::Int)
    (; mass, cache, density_calculator) = system

    resize!(mass, capacity)
    resize_cache!(cache, capacity, density_calculator)
end

resize_cache!(cache, capacity::Int, ::SummationDensity) = resize!(cache.density, capacity)
resize_cache!(cache, capacity::Int, ::ContinuityDensity) = cache

@inline function _wrap_u(_u_cache, system, semi, callback)
    (; ranges_u_cache, nparticles_cache) = callback

    range = ranges_u_cache[system_indices(system, semi)][1]
    n_particles = nparticles_cache[system_indices(system, semi)]

    @boundscheck @assert length(range) == u_nvariables(system) * n_particles

    # This is a non-allocating version of:
    # return unsafe_wrap(Array{eltype(_u_cache), 2}, pointer(view(_u_cache, range)),
    #                    (u_nvariables(system), n_particles))
    return PtrArray(pointer(view(_u_cache, range)),
                    (StaticInt(u_nvariables(system)), n_particles))
end

@inline function _wrap_v(_v_cache, system, semi, callback)
    (; ranges_v_cache, nparticles_cache) = callback

    range = ranges_v_cache[system_indices(system, semi)][1]
    n_particles = nparticles_cache[system_indices(system, semi)]

    @boundscheck @assert length(range) == v_nvariables(system) * n_particles

    # This is a non-allocating version of:
    # return unsafe_wrap(Array{eltype(_v_cache), 2}, pointer(view(_v_cache, range)),
    #                    (v_nvariables(system), n_particles))
    return PtrArray(pointer(view(_v_cache, range)),
                    (StaticInt(v_nvariables(system)), n_particles))
end

function copy_values_v!(v_new, v_old, system, semi, callback)
    (; eachparticle_cache) = callback

    # Copy only unrefined particles
    new_particle_id = 1
    for particle in eachparticle_cache[system_indices(system, semi)]
        for i in 1:v_nvariables(system)
            v_new[i, new_particle_id] = v_old[i, particle]
        end
        new_particle_id += 1
    end
end

function copy_values_u!(u_new, u_old, system, semi, callback)
    (; eachparticle_cache) = callback

    # Copy only unrefined particles
    new_particle_id = 1
    for particle in eachparticle_cache[system_indices(system, semi)]
        for i in 1:u_nvariables(system)
            u_new[i, new_particle_id] = u_old[i, particle]
        end
        new_particle_id += 1
    end
end

@inline get_iterator(system) = eachparticle(system)
@inline get_iterator(system::FluidSystem) = get_iterator(system, system.particle_refinement)

@inline get_iterator(system::FluidSystem, ::Nothing) = eachparticle(system)

@inline function get_iterator(system::FluidSystem,
                              particle_refinement::ParticleRefinement)
    (; candidates) = particle_refinement

    # Filter candidates
    # Uncomment for benchmark
    # return Iterators.filter(i -> !(i in candidates), eachparticle(system))
    return setdiff(eachparticle(system), candidates)
end

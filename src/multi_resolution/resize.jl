function resize!(semi::Semidiscretization, v_ode, u_ode)
    # Preserve non-changing values
    foreach_system(semi) do system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)

        _v = wrap_v(_v_ode, system, semi)
        _u = wrap_u(_u_ode, system, semi)

        eachparticle_ = get_iterator(system)
        copy_values_v!(v, _v, system, eachparticle_)
        copy_values_u!(u, _u, system, eachparticle_)
    end

    # Resize all systems
    foreach_system(semi) do system
        resize!(system, capacity(system))
    end

    # Set ranges after resizing the systems
    ranges_v_, ranges_u_ = ranges_vu(semi.systems)
    for i in 1:length(semi.systems)
        semi.ranges_v[i] = ranges_v_[i]
        semi.ranges_u[i] = ranges_u_[i]
    end

    sizes_v = (v_nvariables(system) * n_moving_particles(system) for system in semi.systems)
    sizes_u = (u_nvariables(system) * n_moving_particles(system) for system in semi.systems)

    # Resize integrated values
    resize!(v_ode, sum(sizes_v))
    resize!(u_ode, sum(sizes_u))

    # TODO: Do the following in the callback
    # resize!(integrator, (length(v_ode), length(u_ode)))

    # # Tell OrdinaryDiffEq that u has been modified
    # u_modified!(integrator, true)

    return semi
end

function resize!(system::WeaklyCompressibleSPHSystem, capacity_system::Int)
    (; mass, pressure, cache, density_calculator) = system

    # TODO: Resize smoothing length
    resize!(mass, capacity_system)
    resize!(pressure, capacity_system)
    resize_cache!(cache, capacity_system, density_calculator)
end

function resize_system!(system::EntropicallyDampedSPHSystem, capacity_system::Int)
    (; mass, cache, density_calculator) = system

    # TODO: Resize smoothing length
    resize!(mass, capacity_system)
    resize_cache!(cache, capacity_system, density_calculator)
end

resize_cache!(cache, n::Int, ::SummationDensity) = resize!(cache.density, n)
resize_cache!(cache, n, ::ContinuityDensity) = cache

function copy_values_v!(v_new, v_old, system, eachparticle_)

    # Copy values before resizing
    new_particle_id = 1
    for particle in eachparticle_
        for i in 1:v_nvariables(system)
            v_new[i, new_particle_id] = v_old[i, particle]
        end
        new_particle_id += 1
    end

    return v_new
end

function copy_values_u!(u_new, u_old, system, eachparticle_)

    # Copy values before resizing
    new_particle_id = 1
    for particle in eachparticle_
        for i in 1:u_nvariables(system)
            u_new[i, new_particle_id] = u_old[i, particle]
        end
        new_particle_id += 1
    end

    return u_new
end

@inline get_iterator(system) = eachparticle(system)

@inline get_iterator(system::FluidSystem) = get_iterator(system, system.particle_refinement)

@inline get_iterator(system::FluidSystem, ::Nothing) = eachparticle(system)

@inline function get_iterator(system::FluidSystem, particle_refinement::ParticleRefinement)
    # TODO
    (candidates_refinement, candidates_coarsening) = particle_refinement

    # Filter candidates
    # Uncomment for benchmark
    # return Iterators.filter(i -> !(i in candidates), eachparticle(system))
    return setdiff(eachparticle(system), candidates_refinement, candidates_coarsening)
end

@inline capacity(system) = capacity(system, system.particle_refinement)

@inline capacity(system, ::Nothing) = nparticles(system)

@inline function capacity(system, particle_refinement)
    (candidates_refine, candidates_coarsen) = particle_refinement

    return nparticles(system) - candidates_refine + candidates_coarsen
end

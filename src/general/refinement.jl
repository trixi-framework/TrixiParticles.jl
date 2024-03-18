# Criteria of refinement:
#
# - fixed (/moving?) refinement zone
# - number of neighbors
# - problem specific criteria (e.g. high velocity gradient)

include("refinement_pattern.jl")

mutable struct ParticleRefinement{RL, NDIMS, ELTYPE, RP, RC}
    candidates          :: Vector{Int}
    refinement_levels   :: Int
    refinement_pattern  :: RP
    refinement_criteria :: RC
    available_childs    :: Int

    # Depends on refinement pattern, particle spacing and parameters ϵ and α.
    # Should be obtained prior to simulation in `create_system_child()`
    rel_position_childs::Vector{SVector{NDIMS, ELTYPE}}

    # It is essential to know the child system, which is empty at the beginning
    # and will be created in `create_system_child()`
    system_child::System

    # API --> parent system with `RL=0`
    function ParticleRefinement(refinement_criteria...;
                                refinement_pattern=CubicSplitting(),
                                refinement_levels=1)
        ELTYPE = eltype(refinement_criteria[1])
        NDIMS = ndims(refinement_criteria[1])

        return new{0, NDIMS, ELTYPE, typeof(refinement_pattern),
                   typeof(refinement_criteria)}([0], refinement_levels, refinement_pattern,
                                                refinement_criteria, 0)
    end

    # Internal constructor for multiple refinement levels
    function ParticleRefinement{RL}(refinement_criteria::Tuple,
                                    refinement_pattern, refinement_levels) where {RL}
        ELTYPE = eltype(refinement_criteria[1])
        NDIMS = ndims(refinement_criteria[1])

        return new{RL, NDIMS, ELTYPE, typeof(refinement_pattern),
                   typeof(refinement_criteria)}([0], refinement_levels, refinement_pattern,
                                                refinement_criteria, 0)
    end
end

@inline Base.ndims(::ParticleRefinement{RL, NDIMS}) where {RL, NDIMS} = ndims

@inline child_set(system, particle_refinement) = Base.OneTo(nchilds(system,
                                                                    particle_refinement))

@inline nchilds(system, ::Nothing) = 0
@inline nchilds(system, pr::ParticleRefinement) = nchilds(system, pr.refinement_pattern)

# ==== Create child systems

function create_system_childs(systems)
    systems_ = ()
    foreach_system(systems) do system
        systems_ = (systems_..., create_system_child(system, system.particle_refinement)...)
    end

    return (systems..., systems_...)
end

create_system_child(system, ::Nothing) = ()

function create_system_child(system::WeaklyCompressibleSPHSystem,
                             particle_refinement::ParticleRefinement{RL}) where {RL}
    (; refinement_levels, refinement_pattern, refinement_criteria) = particle_refinement
    (; density_calculator, state_equation, smoothing_kernel,
    pressure_acceleration_formulation, viscosity, density_diffusion,
    acceleration, correction, source_terms) = system

    NDIMS = ndims(system)

    # Distribute values according to refinement pattern
    smoothing_length_ = smoothing_length_child(system, refinement_pattern)
    particle_refinement.rel_position_childs = refinement_pattern(system)

    # Create "empty" `InitialCondition` for child system
    particle_spacing_ = particle_spacing_child(system, refinement_pattern)
    coordinates_ = zeros(NDIMS, 2)
    velocity_ = similar(coordinates_)
    density_ = system.initial_condition.density[1]
    pressure_ = system.initial_condition.pressure[1]
    mass_ = nothing

    empty_ic = InitialCondition{NDIMS}(coordinates_, velocity_, mass_, density_, pressure_,
                                       particle_spacing_)

    #  Let recursive dispatch handle multiple refinement levels
    level = RL + 1
    particle_refinement_ = if level == refinement_levels
        nothing
    else
        ParticleRefinement{level}(refinement_criteria, refinement_pattern,
                                  refinement_levels)
    end

    system_child = WeaklyCompressibleSPHSystem(empty_ic, density_calculator, state_equation,
                                               smoothing_kernel, smoothing_length_;
                                               pressure_acceleration=pressure_acceleration_formulation,
                                               viscosity, density_diffusion, acceleration,
                                               correction, source_terms,
                                               particle_refinement=particle_refinement_)

    # Empty mass vector leads to `nparticles(system_child) = 0`
    Base.resize!(system_child.mass, 0)

    particle_refinement.system_child = system_child

    return (system_child,
            create_system_child(system_child, system_child.particle_refinement)...)
end

# ==== Refinement
function refinement!(v_ode, u_ode, semi, callback)
    foreach_system(semi) do system
        check_refinement_criteria!(system, v_ode, u_ode, semi)
    end

    resize_and_copy!(callback, semi, v_ode, u_ode)

    refine_particles!(callback, semi, v_ode, u_ode)
end

check_refinement_criteria!(system, v_ode, u_ode, semi) = system

function check_refinement_criteria!(system::FluidSystem, v_ode, u_ode, semi)
    check_refinement_criteria!(system, system.particle_refinement, v_ode, u_ode, semi)
end

check_refinement_criteria!(system, ::Nothing, v_ode, u_ode, semi) = system

function check_refinement_criteria!(system, particle_refinement::ParticleRefinement,
                                    v_ode, u_ode, semi)
    (; candidates, refinement_criteria) = particle_refinement

    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    !isempty(candidates) && Base.resize!(candidates, 0)

    for particle in each_moving_particle(system)
        for refinement_criterion in refinement_criteria
            if (isempty(candidates) || particle != last(candidates)) &&
               refinement_criterion(system, particle, v, u, v_ode, u_ode, semi)
                push!(candidates, particle)
            end
        end
    end

    return system
end

function resize_and_copy!(callback, semi, v_ode, u_ode)
    (; _v_ode, _u_ode) = callback

    # Get non-`resize!`d ranges
    callback.ranges_v_cache, callback.ranges_u_cache = ranges_vu(semi.systems)
    callback.eachparticle_cache = Tuple(get_iterator(system) for system in semi.systems)
    callback.nparticles_cache = Tuple(n_moving_particles(system) for system in semi.systems)

    # Resize internal storage
    Base.resize!(_v_ode, length(v_ode))
    Base.resize!(_u_ode, length(u_ode))

    _v_ode .= v_ode
    _u_ode .= u_ode

    # Resize all systems
    foreach_system(semi) do system
        resize_system!(system, system.particle_refinement)
    end

    # Set `resize!`d ranges
    ranges_v_tmp, ranges_u_tmp = ranges_vu(semi.systems)
    for i in 1:length(semi.systems)
        semi.ranges_v[i][1] = ranges_v_tmp[i][1]
        semi.ranges_u[i][1] = ranges_u_tmp[i][1]
    end

    sizes_u = (u_nvariables(system) * n_moving_particles(system) for system in semi.systems)
    sizes_v = (v_nvariables(system) * n_moving_particles(system) for system in semi.systems)

    # Resize integrated values
    Base.resize!(v_ode, sum(sizes_v))
    Base.resize!(u_ode, sum(sizes_u))

    # Preserve non-changing values
    foreach_system(semi) do system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)
        _v = _wrap_v(_v_ode, system, semi, callback)
        _u = _wrap_u(_u_ode, system, semi, callback)

        copy_values_v!(v, _v, system, semi, callback)
        copy_values_u!(u, _u, system, semi, callback)
    end

    return callback
end

function refine_particles!(callback, semi, v_ode, u_ode)

    # Refine particles in all systems
    foreach_system(semi) do system
        refine_particles!(system, system.particle_refinement, v_ode, u_ode, callback, semi)
    end
end

refine_particles!(system, ::Nothing, v_ode, u_ode, callback, semi) = system

function refine_particles!(system_parent, particle_refinement::ParticleRefinement,
                           v_ode, u_ode, callback, semi)
    (; _v_ode, _u_ode) = callback
    (; candidates, system_child) = particle_refinement

    if !isempty(candidates)
        # Old storage
        v_parent = _wrap_v(_v_ode, system_parent, semi, callback)
        u_parent = _wrap_u(_u_ode, system_parent, semi, callback)

        # Resized storage
        v_child = wrap_v(v_ode, system_child, semi)
        u_child = wrap_u(u_ode, system_child, semi)

        particle_refinement.available_childs = length(candidates) *
                                               nchilds(system_parent, particle_refinement)

        # Loop over all refinement candidates
        for particle_parent in candidates
            bear_childs!(system_child, system_parent, particle_parent, particle_refinement,
                         v_parent, u_parent, v_child, u_child, semi)

            particle_refinement.available_childs -= nchilds(system_parent,
                                                            particle_refinement)
        end
    end
end

# 6 (8) unkowns in 2d (3D) need to be determined for each newly born child particle
# --> mass, position, velocity, smoothing length
#
# Reducing the dof by using a fixed regular refinement pattern
# (given: position and number of child particles)
function bear_childs!(system_child, system_parent, particle_parent, particle_refinement,
                      v_parent, u_parent, v_child, u_child, semi)
    (; rel_position_childs, available_childs, refinement_pattern) = particle_refinement

    nhs = get_neighborhood_search(system_parent, system_parent, semi)
    parent_coords = current_coords(u_parent, system_parent, particle_parent)

    # Loop over all child particles of parent particle
    # The number of child particles depends on the refinement pattern
    for particle_child in child_set(system_parent, particle_refinement)
        absolute_index = particle_child + nparticles(system_child) - available_childs

        # TODO: Handle different masses. Problem: `particle_parent` does not have an
        # mass-entry anymore, since `system_parent` was resized.
        mass_ = system_parent.mass[1]
        system_child.mass[absolute_index] = mass_child(system_parent, mass_,
                                                       refinement_pattern)

        # spread child positions according to the refinement pattern
        child_coords = parent_coords + rel_position_childs[particle_child]
        for dim in 1:ndims(system_child)
            u_child[dim, absolute_index] = child_coords[dim]
        end

        volume = zero(eltype(system_child))
        p_a = zero(eltype(system_child))
        rho_a = zero(eltype(system_child))

        for neighbor in eachneighbor(child_coords, nhs)
            neighbor_coords = current_coords(u_parent, system_parent, neighbor)
            pos_diff = child_coords - neighbor_coords

            distance2 = dot(pos_diff, pos_diff)

            if distance2 <= nhs.search_radius^2
                distance = sqrt(distance2)
                kernel_weight = smoothing_kernel(system_parent, distance)
                volume += kernel_weight

                v_b = current_velocity(v_parent, system_parent, neighbor)
                p_b = particle_pressure_parent(v_parent, system_parent, neighbor)
                rho_b = particle_density_parent(v_parent, system_parent, neighbor)

                for dim in 1:ndims(system_child)
                    v_child[dim, absolute_index] += kernel_weight * v_b[dim]
                end

                rho_a += kernel_weight * rho_b
                p_a += kernel_weight * p_b
            end
        end

        for dim in 1:ndims(system_child)
            v_child[dim, absolute_index] ./ volume
        end

        rho_a /= volume
        p_a /= volume

        set_particle_density(particle_child, v_child, system_child.density_calculator,
                             system_child, rho_a)
        set_particle_pressure(particle_child, v_child, system_child, p_a)
    end

    return system_child
end

@inline resize_system!(system::System, ::Nothing) = system

@inline function resize_system!(system::System, particle_refinement::ParticleRefinement)
    (; candidates, system_child) = particle_refinement

    if !isempty(candidates)
        n_new_child = length(candidates) * nchilds(system, particle_refinement)
        capacity_parent = nparticles(system) - length(candidates)
        capacity_child = nparticles(system_child) + n_new_child

        # Resize child system (extending)
        resize_system!(system_child, capacity_child)

        # Resize parent system (reducing)
        resize_system!(system, capacity_parent)
    end
end

function resize_system!(system::FluidSystem, capacity::Int)
    (; mass, pressure, cache, density_calculator) = system

    Base.resize!(mass, capacity)
    Base.resize!(pressure, capacity)
    resize!(cache, capacity, density_calculator)
end

resize!(cache, capacity::Int, ::SummationDensity) = Base.resize!(cache.density, capacity)
resize!(cache, capacity::Int, ::ContinuityDensity) = cache

@inline function _wrap_u(_u_ode, system, semi, callback)
    (; ranges_u_cache, nparticles_cache) = callback

    range = ranges_u_cache[system_indices(system, semi)][1]
    n_particles = nparticles_cache[system_indices(system, semi)]

    @boundscheck @assert length(range) == u_nvariables(system) * n_particles

    # This is a non-allocating version of:
    # return unsafe_wrap(Array{eltype(_u_ode), 2}, pointer(view(_u_ode, range)),
    #                    (u_nvariables(system), n_particles))
    return PtrArray(pointer(view(_u_ode, range)),
                    (StaticInt(u_nvariables(system)), n_particles))
end

@inline function _wrap_v(_v_ode, system, semi, callback)
    (; ranges_v_cache, nparticles_cache) = callback

    range = ranges_v_cache[system_indices(system, semi)][1]
    n_particles = nparticles_cache[system_indices(system, semi)]

    @boundscheck @assert length(range) == v_nvariables(system) * n_particles

    # This is a non-allocating version of:
    # return unsafe_wrap(Array{eltype(_v_ode), 2}, pointer(view(_v_ode, range)),
    #                    (v_nvariables(system), n_particles))
    return PtrArray(pointer(view(_v_ode, range)),
                    (StaticInt(v_nvariables(system)), n_particles))
end

function copy_values_v!(v_new, v_old, system, semi, callback)
    (; eachparticle_cache) = callback

    # Copy only non-refined particles
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

    # Copy only non-refined particles
    new_particle_id = 1
    for particle in eachparticle_cache[system_indices(system, semi)]
        for i in 1:u_nvariables(system)
            u_new[i, new_particle_id] = u_old[i, particle]
        end
        new_particle_id += 1
    end
end

@inline get_iterator(system) = get_iterator(system, system.particle_refinement)

@inline get_iterator(system, ::Nothing) = eachparticle(system)

@inline function get_iterator(system, particle_refinement::ParticleRefinement)
    (; candidates) = particle_refinement

    # Filter candidates
    #return Iterators.filter(i -> !(i in candidates), eachparticle(system))
    return setdiff(eachparticle(system), candidates)
end

@inline particle_pressure_parent(v, ::WeaklyCompressibleSPHSystem, particle) = 0.0
@inline particle_pressure_parent(v, system::EntropicallyDampedSPHSystem, particle) = particle_pressure(v,
                                                                                                       system,
                                                                                                       particle)

@inline particle_density_parent(v, system, particle) = particle_density_parent(v, system,
                                                                               system.density_calculator,
                                                                               particle)

@inline particle_density_parent(v, system, ::SummationDensity, particle) = 0.0
@inline particle_density_parent(v, system, ::ContinuityDensity, particle) = particle_density(v,
                                                                                             system,
                                                                                             particle)

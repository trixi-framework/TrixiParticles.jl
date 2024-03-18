# Criteria of refinement:
#
# - fixed (/moving?) refinement zone
# - number of neighbors
# - problem specific criteria (e.g. high velocity gradient)

include("refinement_pattern.jl")

mutable struct ParticleRefinement{RL, NDIMS, ELTYPE, RP, RC}
    candidates          :: Vector{ELTYPE}
    refinement_levels   :: Int
    refinement_pattern  :: RP
    refinement_criteria :: RC

    # Depends on refinement pattern, particle spacing and parameters ϵ and α.
    # Should be obtained prior to simulation in `create_system_child()`
    rel_position_childs :: Vector{SVector{NDIMS, ELTYPE}}
    mass_child          :: ELTYPE
    available_childs    :: Int

    # It is essential to know the child system, which is empty at the beginning
    # and will be created in `create_system_child()`
    system_child::System

    # API --> parent system with `RL=0`
    function ParticleRefinement(refinement_criteria...;
                                refinement_pattern=CubicSplitting(),
                                refinement_levels=1)
        ELTYPE = eltype(refinement_criteria)
        NDIMS = ndims(refinement_criteria[1])

        return new{0, NDIMS, ELTYPE, typeof(refinement_pattern),
                   typeof(refinement_criteria)}([], refinement_levels, refinement_pattern,
                                                refinement_criteria)
    end

    # Internal constructor for multiple refinement levels
    function ParticleRefinement{RL}(refinement_criteria::Tuple,
                                    refinement_pattern, refinement_levels) where {RL}
        ELTYPE = eltype(refinement_criteria)
        NDIMS = ndims(refinement_criteria[1])

        return new{RL, NDIMS, ELTYPE, typeof(refinement_pattern),
                   typeof(refinement_criteria)}([], refinement_levels, refinement_pattern,
                                                refinement_criteria)
    end
end

@inline Base.ndims(::ParticleRefinement{RL, NDIMS}) where {RL, NDIMS} = ndims

@inline child_set(system, particle_refinement) = Base.OneTo(nchilds(system,
                                                                    particle_refinement))

@inline ncandidates(::Nothing) = 0
@inline ncandidates(particle_refinement) = particle_refinement.candidates

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
    resize!(system_child.mass, 0)

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

function check_refinement_criteria!(system, particle_refinement::ParticleRefinement,
                                    v_ode, u_ode, semi)
    (; candidates, refinement_criteria) = particle_refinement

    v = wrap_v(v_ode, systm, semi)
    u = wrap_u(u_ode, systm, semi)

    !isempty(candidates) && resize!(candidates, 0)

    for particle in each_moving_particle(system)
        for refinement_criterion in refinement_criteria
            if particle != last(candidates) &&
               refinement_criterion(system, particle, v, u, v_ode, u_ode, semi)
                push!(candidates, particle)
            end
        end
    end

    return system
end

function resize_and_copy!(callback, semi, v_ode, u_ode)
    (; _v_ode, _u_ode, n_candidates, n_childs,
    ranges_v_cache, ranges_u_cache, each_particle_cache) = callback

    # Get non-`resize!`d ranges
    ranges_v_cache, ranges_u_cache = ranges_uv(systems)
    each_particle_cache = Tuple(get_iterator(system) for system in semi.systems)

    # Resize internal storage
    n_total_particles = sum(nparticles.(semi.systems))
    resize!(_v_ode, capacity, n_total_particles)
    resize!(_u_ode, capacity, n_total_particles)

    # Count all candidates for refinement
    n_candidates = 0
    n_childs = 0
    foreach_system(semi) do system
        n_candidates += ncandidates(system.particle_refinement)
        n_childs += ncandidates(system.particle_refinement) *
                    nchilds(system, system.refinement_pattern)
    end

    capacity = n_total_particles - n_candidates + n_childs

    # Resize integrated values
    resize!(v_ode, capacity)
    resize!(u_ode, capacity)

    # Resize all systems
    foreach_system(semi) do system
        resize!(system, system.particle_refinement)
    end

    # Set `resize!`d ranges
    semi.ranges_v, semi.ranges_u = ranges_uv(systems)

    # Preserve non-changing values
    foreach_system(semi) do system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)
        _v = _wrap_v(_v_ode, system, callback)
        _u = _wrap_u(_u_ode, system, callback)

        copy_values_v!(v, _v, system, system.particle_refinement, semi, callback)
        copy_values_u!(u, _u, system, system.particle_refinement, semi, callback)
    end

    return callback
end

refine_particles!(semi::Semidiscretization{Nothing}, v_ode, u_ode) = semi

function refine_particles!(callback, semi, v_ode, u_ode)

    # Refine particles in all systems
    foreach_system(semi) do system
        refine_particles!(system, sysem.particle_refinement, v_ode, u_ode, callback, semi)
    end
end

refine_particles!(system, ::Nothing, v, u, semi) = system

function refine_particles!(system_parent, particle_refinement::ParticleRefinement,
                           v_ode, u_ode, callback, semi)
    (; _v_ode, _u_ode) = callback
    (; candidates, system_child, available_childs) = particle_refinement

    if !isempty(candidates)
        # Old storage
        v_parent = _wrap_v(_v_ode, system_parent, callback)
        u_parent = _wrap_u(_u_ode, system_parent, callback)

        # Resized storage
        v_child = wrap_v(v_ode, system_child, semi)
        u_child = wrap_u(u_ode, system_child, semi)

        available_childs = length(candidates) * nchilds(system_parent, particle_refinement)

        # Loop over all refinement candidates
        for particle_parent in candidates
            bear_childs!(system_child, system_parent, particle_parent, particle_refinement,
                         v_parent, u_parent, v_child, u_child, semi)

            available_childs -= nchilds(system, particle_refinement)
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
    (; rel_position_childs, available_childs, mass_child) = particle_refinement

    nhs = get_neighborhood_search(system_parent, system_parent, semi)
    parent_coords = current_coords(u_parent, system_parent, particle_parent)

    system_child.mass .= mass_child

    # Loop over all child particles of parent particle
    # The number of child particles depends on the refinement pattern
    for particle_child in child_set(system_parent, particle_refinement)
        absolute_index = particle_child + nparticles(system_child) - available_childs

        # spread child positions according to the refinement pattern
        child_coords = parent_coords + rel_position_childs[particle_child]
        for dim in 1:ndims(system_child)
            u_child[absolute_index, dim] = child_coords[dim]
        end

        volume = zero(eltype(system_child))
        p_a = zero(eltype(system_child))
        rho_a = zero(eltype(system_child))

        for neighbor in eachneighbor(child_coords, nhs)
            neighbor_coords = current_coords(u_parent, system_parent, neighbor)
            pos_diff = child_coords - neighbor_coords

            distance2 = dot(pos_diff, pos_diff)

            if distance2 <= search_radius^2
                distance = sqrt(distance2)
                kernel_weight = smoothing_kernel(system_parent, distance)
                volume += kernel_weight

                v_b = current_velocity(v_parent, system_parent, neighbor)
                p_b = particle_pressure(v_parent, system_parent, neighbor)
                rho_b = particle_density(v_parent, system_parent, neighbor)

                for dim in 1:ndims(system_child)
                    v_child[dim, system_child] += kernel_weight * v_b[dim]
                end

                rho_a += kernel_weight * rho_b
                p_a += kernel_weight * p_b
            end
        end

        for dim in 1:ndims(system_child)
            v[dim, system_child] ./ volume
        end

        rho_a /= volume
        p_a /= volume

        set_particle_density(particle_child, v_child, system_child.density_calculator,
                             system_child, rho_a)
        set_particle_pressure(particle_child, v_child, system_child, p_a)
    end

    return system_child
end

@inline Base.resize!(system::System, ::Nothing) = system

@inline function Base.resize!(system::System, particle_refinement::ParticleRefinement)
    (; candidates, system_child) = particle_refinement

    if !isempty(candidates)
        n_new_child = length(candidates) * nchilds(system_parent, particle_refinement)
        capacity_parent = nparticles(system_parent) - length(candidates)
        capacity_child = nparticles(system_child) + n_new_child

        # Resize child system (extending)
        resize!(system_child, capacity_child)

        # Resize parent system (reducing)
        resize!(system_parent, capacity_parent)
    end
end

function Base.resize!(system::WeaklyCompressibleSPHSystem, capacity)
    (; mass, pressure, cache, density_calculator) = system

    resize!(mass, capacity)
    resize!(pressure, capacity)
    resize!(cache, capacity, density_calculator)
end

resize!(cache, capacity, ::SummationDensity) = resize!(cache.density, capacity)
resize!(cache, capacity, ::ContinuityDensity) = cache

@inline function _wrap_u(_u_ode, system, callback)
    (; ranges_u_cache) = callback

    range = ranges_u_cache[system_indices(system, semi)]

    @boundscheck @assert length(range) == u_nvariables(system) * n_moving_particles(system)

    # This is a non-allocating version of:
    # return unsafe_wrap(Array{eltype(_u_ode), 2}, pointer(view(_u_ode, range)),
    #                    (u_nvariables(system), n_moving_particles(system)))
    return PtrArray(pointer(view(_u_ode, range)),
                    (StaticInt(u_nvariables(system)), n_moving_particles(system)))
end

@inline function _wrap_v(_v_ode, system, callback)
    (; ranges_v_cache) = callback

    range = ranges_v_cache[system_indices(system, semi)]

    @boundscheck @assert length(range) == v_nvariables(system) * n_moving_particles(system)

    # This is a non-allocating version of:
    # return unsafe_wrap(Array{eltype(_v_ode), 2}, pointer(view(_v_ode, range)),
    #                    (v_nvariables(system), n_moving_particles(system)))
    return PtrArray(pointer(view(_v_ode, range)),
                    (StaticInt(v_nvariables(system)), n_moving_particles(system)))
end

# `v_new` >= `v_old`
function copy_values_v!(v_new, v_old, system, ::Nothing, semi, callback)
    (; each_particle_cache) = callback

    for particle in each_particle_cache[system_indices(system, semi)]
        for i in 1:v_nvariables(system)
            v_new[i, particle] = v_old[i, particle]
        end
    end
end

# `v_new` <= `v_old`
function copy_values_v!(v_new, v_old, system, particle_refinement::ParticleRefinement,
                        semi, callback)
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

# `u_new` >= `u_old`
function copy_values_u!(u_new, u_old, system, ::Nothing, semi, callback)
    (; each_particle_cache) = callback

    for particle in each_particle_cache[system_indices(system, semi)]
        for i in 1:u_nuariables(system)
            u_new[i, particle] = u_old[i, particle]
        end
    end
end

# `u_new` <= `u_old`
function copy_values_u!(u_new, u_old, system, particle_refinement::ParticleRefinement,
                        semi, callback)
    (; eachparticle_cache) = callback

    # Copy only non-refined particles
    new_particle_id = 1
    for particle in eachparticle_cache[system_indices(system, semi)]
        for i in 1:u_nuariables(system)
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

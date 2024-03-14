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

    # internal `resize!`able storage
    _u :: Vector{ELTYPE}
    _v :: Vector{ELTYPE}

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
function refinement!(system::FluidSystem, v, u, v_ode, u_ode, semi)
    refinement!(system, system.particle_refinement, v, u, v_ode, u_ode, semi)
end

refinement!(system, ::Nothing, v, u, v_ode, u_ode, semi) = system

function refinement!(system, particle_refinement::ParticleRefinement,
                     v, u, v_ode, u_ode, semi)
    check_refinement_criteria!(system, particle_refinement, v, u, v_ode, u_ode, semi)

    refine_particles!(system, particle_refinement, v, u, v_ode, u_ode, semi)
end

function check_refinement_criteria!(system, particle_refinement::ParticleRefinement,
                                    v, u, v_ode, u_ode, semi)
    (; candidates, refinement_criteria) = particle_refinement

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

function refine_particles!(system_parent, particle_refinement::ParticleRefinement,
                           v_parent, u_parent, v_ode, u_ode, semi)
    (; candidates, system_child, _u, _v) = particle_refinement

    if !isempty(candidates)
        n_new_child = length(candidates) * nchilds(system_parent, particle_refinement)
        capacity_parent = nparticles(system_parent) - length(candidates)
        capacity_child = nparticles(system_child) + n_new_child

        particle_refinement.available_childs = n_new_child

        # Resize child system (extending)
        resize!(system_child, capacity_child)

        # Resize internal storage for new child particles
        resize!(_u, n_new_child)
        resize!(_v, n_new_child)

        u_child = PtrArray(pointer(_u),
                           (StaticInt(u_nvariables(system_child)), n_new_child))
        v_child = PtrArray(pointer(_v),
                           (StaticInt(v_nvariables(system_child)), n_new_child))

        # Loop over all refinement candidates
        for particle_parent in candidates
            bear_childs!(system_child, system_parent, particle_parent, particle_refinement,
                         v_parent, u_parent, v_child, u_child, semi)

            particle_refinement.available_childs -= nchilds(system, particle_refinement)
        end

        # Resize parent system (reducing)
        restructure_and_resize!(system_parent, capacity_parent)

        # Resize `v_ode` and `u_ode`
        resize!(v_ode, u_ode, semi, system_parent, system_child,
                capacity_parent, capacity_child)
    end
end

# 6 (8) unkowns in 2d (3D) need to be determined for each newly born child particle
# --> mass, position, velocity, smoothing length
#
# Reducing the dof by using a fixed regular refinement pattern
# (given: position and number of child particles)
function bear_childs!(system_child, system_parent, particle_parent, particle_refinement,
                      v_parent, u_parent, v_child, u_child, semi)
    (; rel_position_childs, available_childs, candidates) = particle_refinement
    n_new_child = length(candidates) * nchilds(system_parent, particle_refinement)

    parent_coords = current_coords(u_parent, system_parent, particle_parent)

    # Loop over all child particles of parent particle
    # The number of child particles depends on the refinement pattern
    for particle_child in child_set(system_parent, particle_refinement)
        absolute_index = particle_child + nparticles(system_child) - available_childs
        relative_index = particle_child + n_new_child - available_childs

        system_child.mass[absolute_index] = particle_refinement.mass_child

        # spread child positions according to the refinement pattern
        child_coords = parent_coords + rel_position_childs[particle_child]
        for dim in 1:ndims(system_child)
            u_child[relative_index, dim] = child_coords[dim]
        end

        # Interpolate field variables
        interpolate_particle_pressure!(system_child, system_parent,
                                       particle_childs, particle_parent,
                                       v_parent, u_parent, v_child, u_child, semi)
        interpolate_particle_density!(system_child, system_parent,
                                      particle_childs, particle_parent,
                                      v_parent, u_parent, v_child, u_child, semi)
        interpolate_current_velocity!(system_child, system_parent,
                                      particle_childs, particle_parent,
                                      v_parent, u_parent, v_child, u_child, semi)
    end

    return system_child
end

function Base.resize!(system::WeaklyCompressibleSPHSystem, capacity)
    (; mass, pressure, cache, density_calculator) = system

    resize!(mass, capacity)
    resize!(pressure, capacity)
    resize!(cache, capacity, density_calculator)
end

resize!(cache, capacity, ::SummationDensity) = resize!(cache.density, capacity)
resize!(cache, capacity, ::ContinuityDensity) = cache

function resize!(v_ode, u_ode, semi, system_parent, system_child,
                 capacity_parent, capacity_child)
    (; particle_refinement) = system_parent
    (; candidates) = particle_refinement
end

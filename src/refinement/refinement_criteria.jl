abstract type AbstractRefinementCriterion end

"""
    SpatialRefinementCriterion()

Refinement criterion that sets the target spacing of fluid particles near boundaries
to the particle spacing of the boundary.
A fluid particle is near a boundary if a boundary particle is within the compact support
of the fluid particle or of the boundary particle.
This applies to `WallBoundarySystem`s, structures and `OpenBoundarySystem`s.
"""
struct SpatialRefinementCriterion <: AbstractRefinementCriterion end

"""
    SolutionRefinementCriterion(target_spacing)

Refinement criterion that sets the target spacing of fluid particles based on the solution.
The function `target_spacing(system, particle, v, u, semi)` returns the requested target
spacing of `particle`. The criterion is only active where the requested spacing is smaller
than `max_spacing` of the [`ParticleRefinement`](@ref).

A solution criterion can only decrease the target spacing,
and it does not apply to particles near boundaries (see [`SpatialRefinementCriterion`](@ref)).
With multiple solution criteria, the smallest requested spacing is used.
"""
struct SolutionRefinementCriterion{F} <: AbstractRefinementCriterion
    target_spacing::F
end

# Set the target spacing of the particles in the regions of the refinement criteria
# and mark them as fixed for the target spacing update (Alg. 4 in Haftu et al.).
@inline apply_refinement_criteria!(system, v_ode, u_ode, semi) = system

@inline function apply_refinement_criteria!(system::AbstractFluidSystem, v_ode, u_ode, semi)
    return apply_refinement_criteria!(system, get_refinement(system), v_ode, u_ode, semi)
end

@inline apply_refinement_criteria!(system, ::Nothing, v_ode, u_ode, semi) = system

function apply_refinement_criteria!(system, refinement, v_ode, u_ode, semi)
    (; criteria) = refinement

    # The target spacing persists between refinement calls and is not reset here.
    # Only reset which particles are fixed in this call.
    set_zero!(system.cache.is_fixed)

    # The spatial criteria run first, since the solution criteria skip the particles
    # that the spatial criteria fixed (`MarkBoundary` before `SolutionAdaptive`
    # in the code of Haftu et al.).
    if any(criterion -> criterion isa SpatialRefinementCriterion, criteria)
        foreach_system(semi) do neighbor_system
            apply_spatial_criterion!(system, neighbor_system, v_ode, u_ode, semi)
        end
    end

    apply_solution_criteria!(system, criteria, v_ode, u_ode, semi)

    return system
end

@inline apply_spatial_criterion!(system, neighbor_system, v_ode, u_ode, semi) = system

function apply_spatial_criterion!(system,
                                  neighbor_system::Union{WallBoundarySystem,
                                                         AbstractStructureSystem,
                                                         OpenBoundarySystem},
                                  v_ode, u_ode, semi)
    (; target_spacing, is_fixed) = system.cache
    neighbor_spacing = neighbor_system.initial_condition.particle_spacing

    u = wrap_u(u_ode, system, semi)
    u_neighbor = wrap_u(u_ode, neighbor_system, semi)
    system_coords = current_coordinates(u, system)
    neighbor_coords = current_coordinates(u_neighbor, neighbor_system)

    foreach_symmetric_point_neighbor(system, neighbor_system,
                                     system_coords, neighbor_coords, semi;
                                     points=each_integrated_particle(system)) do particle,
                                                                                 _, _, _
        # The first boundary in this call overwrites the target spacing of the previous
        # call, further boundaries can only decrease it (`MarkBoundary` in the code
        # of Haftu et al.). Note that each thread only writes to its own `particle`.
        if is_fixed[particle]
            target_spacing[particle] = min(target_spacing[particle], neighbor_spacing)
        else
            target_spacing[particle] = neighbor_spacing
            is_fixed[particle] = true
        end
    end

    return system
end

function apply_solution_criteria!(system, criteria, v_ode, u_ode, semi)
    (; target_spacing, is_fixed) = system.cache
    (; max_spacing) = get_refinement(system)

    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    @threaded semi for particle in each_integrated_particle(system)
        # Particles near boundaries keep the boundary spacing
        # (`SolutionAdaptive` in the code of Haftu et al.)
        is_fixed[particle] && return

        requested_spacing = mapreduce(min, criteria) do criterion
            return requested_target_spacing(criterion, system, particle, v, u, semi)
        end

        # The criteria are only active where they request a spacing below `max_spacing`
        requested_spacing < max_spacing || return

        # A solution criterion can only decrease the spacing of the previous call
        target_spacing[particle] = min(target_spacing[particle], requested_spacing)
        is_fixed[particle] = true
    end

    return system
end

# Spatial criteria are applied in `apply_spatial_criterion!`
@inline function requested_target_spacing(::SpatialRefinementCriterion, system, particle,
                                          v, u, semi)
    return typemax(eltype(system))
end

@inline function requested_target_spacing(criterion::SolutionRefinementCriterion, system,
                                          particle, v, u, semi)
    return criterion.target_spacing(system, particle, v, u, semi)
end

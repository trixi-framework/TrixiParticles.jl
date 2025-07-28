abstract type RefinementCriteria end

struct SpatialRefinementCriterion <: RefinementCriteria end

struct SolutionRefinementCriterion <: RefinementCriteria end

function check_refinement_criteria!(semi, v_ode, u_ode)
    foreach_system(semi) do system
        check_refinement_criteria!(system, v_ode, u_ode, semi)
    end
end

@inline check_refinement_criteria!(system, v_ode, u_ode, semi) = system

@inline function check_refinement_criteria!(system::FluidSystem, v_ode, u_ode, semi)
    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    check_refinement_criteria!(system, system.particle_refinement, v, u, v_ode, u_ode, semi)
end

@inline function check_refinement_criteria!(system::FluidSystem, ::Nothing,
                                            v, u, v_ode, u_ode, semi)
    system
end

@inline function check_refinement_criteria!(system::FluidSystem, refinement,
                                            v, u, v_ode, u_ode, semi)
    (; refinement_criteria) = system.particle_refinement
    for criterion in refinement_criteria
        criterion(system, v, u, v_ode, u_ode, semi)
    end
end

@inline function (criterion::SpatialRefinementCriterion)(system, v, u, v_ode, u_ode, semi)
    system_coords = current_coordinates(u, system)

    foreach_system(semi) do neighbor_system
        set_particle_spacing!(system, neighbor_system, system_coords, v_ode, u_ode, semi)
    end
    return system
end

@inline set_particle_spacing!(system, _, _, _, _, _) = system

@inline function set_particle_spacing!(particle_system,
                                       neighbor_system::Union{BoundarySystem, SolidSystem},
                                       system_coords, v_ode, u_ode, semi)
    (; smoothing_length, smoothing_length_factor) = particle_system.cache

    u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    foreach_point_neighbor(particle_system, neighbor_system,
                           system_coords, neighbor_coords,
                           semi) do particle, neighbor, pos_diff, distance
        # Only consider particles with a distance > 0.
        distance < sqrt(eps()) && return

        dp_particle = particle_spacing(particle_system, particle)
        dp_neighbor = particle_spacing(neighbor_system, neighbor)

        smoothing_length[particle] = smoothing_length_factor * min(dp_neighbor, dp_particle)
    end

    return particle_system
end

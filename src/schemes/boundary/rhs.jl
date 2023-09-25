# Interaction of boundary  with other systems
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system, neighborhood_search,
                   particle_system::BoundarySPHSystem,
                   neighbor_system)
    # TODO Solids and moving boundaries should be considered in the continuity equation
    return dv
end

# Boundary-fluid interaction with dummy particles model
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system, neighborhood_search,
                   particle_system::BoundarySPHSystem{<:BoundaryModelDummyParticles},
                   neighbor_system::FluidSystem)
    (; density_calculator) = particle_system.boundary_model

    interact!(dv, v_particle_system, u_particle_system,
              v_neighbor_system, u_neighbor_system, neighborhood_search,
              particle_system, neighbor_system, density_calculator)
end

function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system, neighborhood_search,
                   particle_system::BoundarySPHSystem, neighbor_system, density_calculator)
    return dv
end

# With `ContinuityDensity` solve the continuity equation
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system, neighborhood_search,
                   particle_system::BoundarySPHSystem, neighbor_system, ::ContinuityDensity)
    (; boundary_model) = particle_system

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    for_particle_neighbor(particle_system, neighbor_system,
                          system_coords, neighbor_coords,
                          neighborhood_search) do particle, neighbor, pos_diff, distance
        # Continuity equation
        vdiff = current_velocity(v_particle_system, particle_system, particle) -
                current_velocity(v_neighbor_system, neighbor_system, neighbor)

        # For boundary particles, the velocity is not integrated.
        # Therefore, the density is stored in the first dimension of `dv`.
        dv[1, particle] += sum(neighbor_system.mass[neighbor] * vdiff .*
                               smoothing_kernel_grad(boundary_model, pos_diff,
                                                     distance))
    end

    return dv
end

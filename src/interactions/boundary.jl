# Boundary-fluid interaction
function interact!(dv, v_particle_container, u_particle_container,
                   v_neighbor_container, u_neighbor_container, neighborhood_search,
                   particle_container::BoundaryParticleContainer,
                   neighbor_container)
    @unpack boundary_model = particle_container

    interact!(dv, v_particle_container, u_particle_container,
              v_neighbor_container, u_neighbor_container, neighborhood_search,
              particle_container, neighbor_container, boundary_model)
end

function interact!(dv, v_particle_container, u_particle_container,
                   v_neighbor_container, u_neighbor_container, neighborhood_search,
                   particle_container::BoundaryParticleContainer,
                   neighbor_container::BoundaryParticleContainer)
    # TODO moving boundaries
    return dv
end

function interact!(dv, v_particle_container, u_particle_container,
                   v_neighbor_container, u_neighbor_container, neighborhood_search,
                   particle_container::BoundaryParticleContainer,
                   neighbor_container,
                   boundary_model)
    return dv
end

function interact!(dv, v_particle_container, u_particle_container,
                   v_neighbor_container, u_neighbor_container, neighborhood_search,
                   particle_container::BoundaryParticleContainer,
                   neighbor_container,
                   boundary_model::BoundaryModelDummyParticles)
    @unpack density_calculator = boundary_model

    interact!(dv, v_particle_container, u_particle_container,
              v_neighbor_container, u_neighbor_container, neighborhood_search,
              particle_container, neighbor_container, boundary_model, density_calculator)
end

function interact!(dv, v_particle_container, u_particle_container,
                   v_neighbor_container, u_neighbor_container, neighborhood_search,
                   particle_container::BoundaryParticleContainer,
                   neighbor_container, boundary_model,
                   density_calculator)
    return dv
end

function interact!(dv, v_particle_container, u_particle_container,
                   v_neighbor_container, u_neighbor_container, neighborhood_search,
                   particle_container::BoundaryParticleContainer,
                   neighbor_container, boundary_model,
                   ::ContinuityDensity)
    @unpack smoothing_kernel, smoothing_length = boundary_model

    @threaded for particle in each_moving_particle(particle_container)
        particle_coords = get_current_coords(particle, u_particle_container,
                                             particle_container)
        for neighbor in eachneighbor(particle_coords, neighborhood_search)
            neighbor_coords = get_current_coords(neighbor, u_neighbor_container,
                                                 neighbor_container)

            pos_diff = particle_coords - neighbor_coords
            distance2 = dot(pos_diff, pos_diff)

            if eps() < distance2 <= compact_support(boundary_model)^2
                distance = sqrt(distance2)

                # Continuity equation
                vdiff = get_particle_vel(particle, v_particle_container,
                                         particle_container) -
                        get_particle_vel(neighbor, v_neighbor_container, neighbor_container)

                # For boundary particles, the velocity is not integrated.
                # Therefore, the density is stored in the first dimension of `dv`.
                dv[1, particle] += sum(neighbor_container.mass[neighbor] * vdiff .*
                                       kernel_grad(smoothing_kernel, pos_diff, distance,
                                                   smoothing_length))
            end
        end
    end

    return dv
end

# Boundary-fluid interaction
function interact!(du, u_particle_container, u_neighbor_container, neighborhood_search,
                   particle_container::BoundaryParticleContainer,
                   neighbor_container)
    @unpack boundary_model = particle_container

    interact!(du, u_particle_container, u_neighbor_container, neighborhood_search,
              particle_container, neighbor_container, boundary_model)
end

function interact!(du, u_particle_container, u_neighbor_container, neighborhood_search,
                   particle_container::BoundaryParticleContainer,
                   neighbor_container::BoundaryParticleContainer)
    # TODO moving boundaries
    return du
end

function interact!(du, u_particle_container, u_neighbor_container, neighborhood_search,
                   particle_container::BoundaryParticleContainer,
                   neighbor_container,
                   boundary_model)
    return du
end

function interact!(du, u_particle_container, u_neighbor_container, neighborhood_search,
                   particle_container::BoundaryParticleContainer,
                   neighbor_container,
                   boundary_model::BoundaryModelDummyParticles)
    @unpack density_calculator = boundary_model

    interact!(du, u_particle_container, u_neighbor_container, neighborhood_search,
              particle_container, neighbor_container, boundary_model, density_calculator)
end

function interact!(du, u_particle_container, u_neighbor_container, neighborhood_search,
                   particle_container::BoundaryParticleContainer,
                   neighbor_container, boundary_model,
                   density_calculator)
    return du
end

function interact!(du, u_particle_container, u_neighbor_container, neighborhood_search,
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
            distance = norm(pos_diff)

            if sqrt(eps()) < distance <= compact_support(smoothing_kernel, smoothing_length)
                # Continuity equation
                vdiff = get_particle_vel(particle, u_particle_container,
                                         particle_container) -
                        get_particle_vel(neighbor, u_neighbor_container, neighbor_container)

                du[1, particle] += sum(neighbor_container.mass[neighbor] * vdiff *
                                       kernel_deriv(smoothing_kernel, distance,
                                                    smoothing_length) .*
                                       pos_diff) / distance
            end
        end
    end

    return du
end

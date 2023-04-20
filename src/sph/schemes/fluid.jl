include("WCSPH.jl")

# Nothing to initialize for this container
initialize!(container::FluidParticleContainer, neighborhood_search) = container

function update!(container::FluidParticleContainer, container_index, v, u, v_ode, u_ode,
                 semi, t)
    @unpack density_calculator = container

    compute_quantities(container, container_index, density_calculator, v, u, u_ode, semi, t)

    return container
end

function comput_density!(container::FluidParticleContainer, container_index,
                         ::SummationDensity, v, u, u_ode, semi)
    @unpack particle_containers, neighborhood_searches = semi
    @unpack cache = container
    @unpack density = cache # Density is in the cache for SummationDensity

    density .= zero(eltype(density))

    # Use all other containers for the density summation
    @trixi_timeit timer() "compute density" foreach_enumerate(particle_containers) do (neighbor_container_index,
                                                                                       neighbor_container)
        u_neighbor_container = wrap_u(u_ode, neighbor_container_index,
                                      neighbor_container, semi)

        @threaded for particle in eachparticle(container)
            compute_density_per_particle(particle, u, u_neighbor_container,
                                         container, neighbor_container,
                                         neighborhood_searches[container_index][neighbor_container_index])
        end
    end
end

# Use this function barrier and unpack inside to avoid passing closures to Polyester.jl with @batch (@threaded).
# Otherwise, @threaded does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
@inline function compute_density_per_particle(particle,
                                              u_particle_container, u_neighbor_container,
                                              particle_container::FluidParticleContainer,
                                              neighbor_container, neighborhood_search)
    @unpack smoothing_kernel, smoothing_length, cache = particle_container
    @unpack density = cache # Density is in the cache for SummationDensity

    particle_coords = get_current_coords(particle, u_particle_container, particle_container)
    for neighbor in eachneighbor(particle_coords, neighborhood_search)
        mass = get_hydrodynamic_mass(neighbor, neighbor_container)
        neighbor_coords = get_current_coords(neighbor, u_neighbor_container,
                                             neighbor_container)
        distance = norm(particle_coords - neighbor_coords)

        if distance <= compact_support(smoothing_kernel, smoothing_length)
            density[particle] += mass * kernel(smoothing_kernel, distance, smoothing_length)
        end
    end
end

function comput_density!(container::FluidParticleContainer, container_index,
                         ::ContinuityDensity, v, u, u_ode, semi)
end

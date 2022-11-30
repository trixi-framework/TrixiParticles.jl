"""
    FluidParticleContainer(particle_coordinates, particle_velocities, particle_masses,
                           density_calculator::SummationDensity, state_equation,
                           smoothing_kernel, smoothing_length;
                           viscosity=NoViscosity(),
                           acceleration=ntuple(_ -> 0.0, size(particle_coordinates, 1)))

    FluidParticleContainer(particle_coordinates, particle_velocities, particle_masses, particle_densities,
                           density_calculator::ContinuityDensity, state_equation,
                           smoothing_kernel, smoothing_length;
                           viscosity=NoViscosity(),
                           acceleration=ntuple(_ -> 0.0, size(particle_coordinates, 1)))

Container for fluid particles. With [`ContinuityDensity`](@ref), the `particle_densities` array has to be passed.
"""
struct FluidParticleContainer{NDIMS, ELTYPE<:Real, DC, SE, K, V, C} <: ParticleContainer{NDIMS}
    initial_coordinates ::Array{ELTYPE, 2} # [dimension, particle]
    initial_velocity    ::Array{ELTYPE, 2} # [dimension, particle]
    mass                ::Array{ELTYPE, 1} # [particle]
    pressure            ::Array{ELTYPE, 1} # [particle]
    density_calculator  ::DC
    state_equation      ::SE
    smoothing_kernel    ::K
    smoothing_length    ::ELTYPE
    viscosity           ::V
    acceleration        ::SVector{NDIMS, ELTYPE}
    cache               ::C

    function FluidParticleContainer(particle_coordinates, particle_velocities, particle_masses,
                                    density_calculator::SummationDensity, state_equation,
                                    smoothing_kernel, smoothing_length;
                                    viscosity=NoViscosity(),
                                    acceleration=ntuple(_ -> 0.0, size(particle_coordinates, 1)))
        NDIMS = size(particle_coordinates, 1)
        ELTYPE = eltype(particle_masses)
        nparticles = length(particle_masses)

        pressure = Vector{ELTYPE}(undef, nparticles)

        # Make acceleration an SVector
        acceleration_ = SVector(acceleration...)

        density = Vector{ELTYPE}(undef, nparticles)
        cache = (; density)

        return new{NDIMS, ELTYPE, typeof(density_calculator), typeof(state_equation),
                   typeof(smoothing_kernel), typeof(viscosity), typeof(cache)}(
            particle_coordinates, particle_velocities, particle_masses, pressure,
            density_calculator, state_equation, smoothing_kernel, smoothing_length,
            viscosity, acceleration_, cache)
    end

    function FluidParticleContainer(particle_coordinates, particle_velocities, particle_masses, particle_densities,
                                    density_calculator::ContinuityDensity, state_equation,
                                    smoothing_kernel, smoothing_length;
                                    viscosity=NoViscosity(),
                                    acceleration=ntuple(_ -> 0.0, size(particle_coordinates, 1)))
        NDIMS = size(particle_coordinates, 1)
        ELTYPE = eltype(particle_masses)
        nparticles = length(particle_masses)

        pressure = Vector{ELTYPE}(undef, nparticles)

        # Make acceleration an SVector
        acceleration_ = SVector(acceleration...)

        initial_density = particle_densities
        cache = (; initial_density)

        return new{NDIMS, ELTYPE, typeof(density_calculator), typeof(state_equation),
                   typeof(smoothing_kernel), typeof(viscosity), typeof(cache)}(
            particle_coordinates, particle_velocities, particle_masses, pressure,
            density_calculator, state_equation, smoothing_kernel, smoothing_length,
            viscosity, acceleration_, cache)
    end
end


@inline nvariables(container::FluidParticleContainer) = nvariables(container, container.density_calculator)
@inline nvariables(container, ::SummationDensity) = 2 * ndims(container)
@inline nvariables(container, ::ContinuityDensity) = 2 * ndims(container) + 1


# Nothing to initialize for this container
initialize!(container::FluidParticleContainer, neighborhood_search) = container


function update!(container::FluidParticleContainer, u, u_ode, neighborhood_search, semi)
    @unpack density_calculator = container

    compute_quantities(u, density_calculator, container, u_ode, semi)
end


function compute_quantities(u, ::ContinuityDensity, container, u_ode, semi)
    compute_pressure!(container, u)
end

function compute_quantities(u, ::SummationDensity, container, u_ode, semi)
    @unpack particle_containers = semi
    @unpack cache = container
    @unpack density = cache # Density is in the cache for SummationDensity

    density .= zero(eltype(density))

    # Use all other containers for the density summation
    @pixie_timeit timer() "compute density" for (neighbor_container_index, neighbor_container) in pairs(particle_containers)
        u_neighbor_container = wrap_array(u_ode, neighbor_container_index, semi)

        @threaded for particle in eachparticle(container)
            compute_density_per_particle(particle, u, u_neighbor_container,
                                         container, neighbor_container)
        end
    end

    compute_pressure!(container, u)
end


# Use this function barrier and unpack inside to avoid passing closures to Polyester.jl with @batch (@threaded).
# Otherwise, @threaded does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
@inline function compute_density_per_particle(particle, u_particle_container, u_neighbor_container,
                                              particle_container, neighbor_container)
    @unpack smoothing_kernel, smoothing_length, cache = particle_container
    @unpack density = cache # Density is in the cache for SummationDensity
    @unpack mass = neighbor_container

    particle_coords = get_current_coords(particle, u_particle_container, particle_container)
    for neighbor in eachneighbor(particle_coords, neighbor_container)
        distance = norm(particle_coords - get_current_coords(neighbor, u_neighbor_container, neighbor_container))

        if distance <= compact_support(smoothing_kernel, smoothing_length)
            density[particle] += mass[neighbor] * kernel(smoothing_kernel, distance, smoothing_length)
        end
    end
end


function compute_pressure!(container, u)
    @unpack state_equation, pressure = container

    # Note that @threaded makes this slower
    for particle in eachparticle(container)
        pressure[particle] = state_equation(get_particle_density(particle, u, container))
    end
end


function write_variables!(u0, container::FluidParticleContainer)
    @unpack initial_coordinates, initial_velocity, density_calculator = container

    for particle in eachparticle(container)
        # Write particle coordinates
        for dim in 1:ndims(container)
            u0[dim, particle] = initial_coordinates[dim, particle]
        end

        # Write particle velocities
        for dim in 1:ndims(container)
            u0[dim + ndims(container), particle] = initial_velocity[dim, particle]
        end
    end

    write_variables!(u0, density_calculator, container)

    return u0
end


function write_variables!(u0, ::SummationDensity, container)
    return u0
end

function write_variables!(u0, ::ContinuityDensity, container)
    @unpack cache = container
    @unpack initial_density = cache

    for particle in eachparticle(container)
        # Set particle densities
        u0[2 * ndims(container) + 1, particle] = initial_density[particle]
    end

    return u0
end

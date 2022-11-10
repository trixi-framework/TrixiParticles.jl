struct FluidParticleContainer{NDIMS, ELTYPE<:Real, DC, SE, K, V, NS, C} <: ParticleContainer{NDIMS}
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
    neighborhood_search ::NS
    cache               ::C

    function FluidParticleContainer(particle_coordinates, particle_velocities, particle_masses,
                                    density_calculator::SummationDensity, state_equation,
                                    smoothing_kernel, smoothing_length;
                                    viscosity=NoViscosity(),
                                    acceleration=ntuple(_ -> 0.0, size(particle_coordinates, 1)),
                                    neighborhood_search=nothing)
        NDIMS = size(particle_coordinates, 1)
        ELTYPE = eltype(particle_masses)
        nparticles = length(particle_masses)

        pressure = Vector{ELTYPE}(undef, nparticles)

        # Make acceleration an SVector
        acceleration_ = SVector(acceleration...)

        density = Vector{ELTYPE}(undef, nparticles)
        cache = (; density)

        return new{NDIMS, ELTYPE, typeof(density_calculator), typeof(state_equation),
                   typeof(smoothing_kernel), typeof(viscosity),
                   typeof(neighborhood_search), typeof(cache)}(
            particle_coordinates, particle_velocities, particle_masses, pressure,
            density_calculator, state_equation, smoothing_kernel, smoothing_length,
            viscosity, acceleration_, neighborhood_search, cache)
    end

    function FluidParticleContainer(particle_coordinates, particle_velocities, particle_masses, particle_densities,
                            density_calculator::ContinuityDensity, state_equation,
                            smoothing_kernel, smoothing_length;
                            viscosity=NoViscosity(),
                            acceleration=ntuple(_ -> 0.0, size(particle_coordinates, 1)),
                            neighborhood_search=nothing)
        NDIMS = size(particle_coordinates, 1)
        ELTYPE = eltype(particle_masses)
        nparticles = length(particle_masses)

        pressure = Vector{ELTYPE}(undef, nparticles)

        # Make acceleration an SVector
        acceleration_ = SVector(acceleration...)

        initial_density = particle_densities
        cache = (; initial_density)

        return new{NDIMS, ELTYPE, typeof(density_calculator), typeof(state_equation),
                   typeof(smoothing_kernel), typeof(viscosity),
                   typeof(neighborhood_search), typeof(cache)}(
            particle_coordinates, particle_velocities, particle_masses, pressure,
            density_calculator, state_equation, smoothing_kernel, smoothing_length,
            viscosity, acceleration_, neighborhood_search, cache)
    end
end


@inline nvariables(container::FluidParticleContainer) = nvariables(container, container.density_calculator)
@inline nvariables(container, ::SummationDensity) = 2 * ndims(container)
@inline nvariables(container, ::ContinuityDensity) = 2 * ndims(container) + 1


function initialize!(container::FluidParticleContainer)
    @unpack initial_coordinates, neighborhood_search = container

    # Initialize neighborhood search
    @pixie_timeit timer() "initialize neighborhood search" initialize!(neighborhood_search, initial_coordinates, container)
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

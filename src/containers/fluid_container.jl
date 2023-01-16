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
struct FluidParticleContainer{NDIMS, ELTYPE<:Real, DC, SE, K, V, C, ST} <: ParticleContainer{NDIMS}
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
    surface_tension     ::ST

    function FluidParticleContainer(particle_coordinates, particle_velocities,
                                    particle_masses,
                                    density_calculator::SummationDensity, state_equation,
                                    smoothing_kernel, smoothing_length;
                                    viscosity=NoViscosity(),
                                    acceleration=ntuple(_ -> 0.0, size(particle_coordinates, 1)),
                                    surface_tension=NoSurfaceTension())

        NDIMS = size(particle_coordinates, 1)
        ELTYPE = eltype(particle_masses)
        nparticles = length(particle_masses)

        pressure = Vector{ELTYPE}(undef, nparticles)

        # Make acceleration an SVector
        acceleration_ = SVector(acceleration...)
        if length(acceleration_) != NDIMS
            error("Acceleration must be of length $NDIMS for a $(NDIMS)D problem")
        end

        density = Vector{ELTYPE}(undef, nparticles)
        cache = (; density)

        return new{NDIMS, ELTYPE, typeof(density_calculator), typeof(state_equation),
                   typeof(smoothing_kernel), typeof(viscosity), typeof(cache), typeof(surface_tension)}(
            particle_coordinates, particle_velocities, particle_masses, pressure,
            density_calculator, state_equation, smoothing_kernel, smoothing_length,
            viscosity, acceleration_, cache, surface_tension)
    end

    function FluidParticleContainer(particle_coordinates, particle_velocities,
                                    particle_masses, particle_densities,
                                    density_calculator::ContinuityDensity, state_equation,
                                    smoothing_kernel, smoothing_length;
                                    viscosity=NoViscosity(),
                                    acceleration=ntuple(_ -> 0.0, size(particle_coordinates, 1)),
                                    surface_tension=NoSurfaceTension())

        NDIMS = size(particle_coordinates, 1)
        ELTYPE = eltype(particle_masses)
        nparticles = length(particle_masses)

        pressure = Vector{ELTYPE}(undef, nparticles)

        # Make acceleration an SVector
        acceleration_ = SVector(acceleration...)
        if length(acceleration_) != NDIMS
            error("Acceleration must be of length $NDIMS for a $(NDIMS)D problem")
        end

        initial_density = particle_densities
        cache = (; initial_density)

        return new{NDIMS, ELTYPE, typeof(density_calculator), typeof(state_equation),
                   typeof(smoothing_kernel), typeof(viscosity), typeof(cache), typeof(surface_tension)}(
            particle_coordinates, particle_velocities, particle_masses, pressure,
            density_calculator, state_equation, smoothing_kernel, smoothing_length,
            viscosity, acceleration_, cache, surface_tension)
    end
end

function Base.show(io::IO, container::FluidParticleContainer)
    @nospecialize container # reduce precompilation time

    print(io, "FluidParticleContainer{", ndims(container), "}(")
    print(io, container.density_calculator)
    print(io, ", ", container.state_equation)
    print(io, ", ", container.smoothing_kernel)
    print(io, ", ", container.viscosity)
    print(io, ", ", container.acceleration)
    print(io, ") with ", nparticles(container), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", container::FluidParticleContainer)
    @nospecialize container # reduce precompilation time

    if get(io, :compact, false)
        show(io, container)
    else
        summary_header(io, "FluidParticleContainer{$(ndims(container))}")
        summary_line(io, "#particles", nparticles(container))
        summary_line(io, "density calculator",
                     container.density_calculator |> typeof |> nameof)
        summary_line(io, "state equation", container.state_equation |> typeof |> nameof)
        summary_line(io, "smoothing kernel", container.smoothing_kernel |> typeof |> nameof)
        summary_line(io, "viscosity", container.viscosity)
        summary_line(io, "acceleration", container.acceleration)
        summary_line(io, "surface tension", container.surface_tension)
        summary_footer(io)
    end
end

@inline function nvariables(container::FluidParticleContainer)
    nvariables(container, container.density_calculator)
end
@inline function nvariables(container::FluidParticleContainer, ::SummationDensity)
    2 * ndims(container)
end
@inline function nvariables(container::FluidParticleContainer, ::ContinuityDensity)
    2 * ndims(container) + 1
end

# Nothing to initialize for this container
initialize!(container::FluidParticleContainer, neighborhood_search) = container

function update!(container::FluidParticleContainer, container_index, u, u_ode, semi, t)
    @unpack density_calculator = container

    compute_quantities(u, density_calculator, container, container_index, u_ode, semi)

    return container
end

function compute_quantities(u, ::ContinuityDensity, container, container_index, u_ode, semi)
    compute_pressure!(container, u)
end

function compute_quantities(u, ::SummationDensity, container, container_index, u_ode, semi)
    @unpack particle_containers, neighborhood_searches = semi
    @unpack cache = container
    @unpack density = cache # Density is in the cache for SummationDensity

    density .= zero(eltype(density))

    # Use all other containers for the density summation
    @pixie_timeit timer() "compute density" foreach_enumerate(particle_containers) do (neighbor_container_index,
                                                                                       neighbor_container)
        u_neighbor_container = wrap_array(u_ode, neighbor_container_index,
                                          neighbor_container, semi)

        @threaded for particle in eachparticle(container)
            compute_density_per_particle(particle, u, u_neighbor_container,
                                         container, neighbor_container,
                                         neighborhood_searches[container_index][neighbor_container_index])
        end
    end

    compute_pressure!(container, u)
end

# Use this function barrier and unpack inside to avoid passing closures to Polyester.jl with @batch (@threaded).
# Otherwise, @threaded does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
@inline function compute_density_per_particle(particle, u_particle_container,
                                              u_neighbor_container,
                                              particle_container::FluidParticleContainer,
                                              neighbor_container, neighborhood_search)
    @unpack smoothing_kernel, smoothing_length, cache = particle_container
    @unpack density = cache # Density is in the cache for SummationDensity
    @unpack mass = neighbor_container

    particle_coords = get_current_coords(particle, u_particle_container, particle_container)
    for neighbor in eachneighbor(particle_coords, neighborhood_search)
        distance = norm(particle_coords -
                        get_current_coords(neighbor, u_neighbor_container,
                                           neighbor_container))

        if distance <= compact_support(smoothing_kernel, smoothing_length)
            density[particle] += mass[neighbor] *
                                 kernel(smoothing_kernel, distance, smoothing_length)
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

function write_variables!(u0, ::SummationDensity, container::FluidParticleContainer)
    return u0
end

function write_variables!(u0, ::ContinuityDensity, container::FluidParticleContainer)
    @unpack cache = container
    @unpack initial_density = cache

    for particle in eachparticle(container)
        # Set particle densities
        u0[2 * ndims(container) + 1, particle] = initial_density[particle]
    end

    return u0
end

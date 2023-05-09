"""
    FluidParticleContainer(setup,
                           density_calculator, state_equation,
                           smoothing_kernel, smoothing_length;
                           viscosity=NoViscosity(),
                           acceleration=ntuple(_ -> 0.0, size(particle_coordinates, 1)))
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
struct FluidParticleContainer{NDIMS, ELTYPE <: Real, DC, SE, K, V, C} <:
       ParticleContainer{NDIMS}
    initial_coordinates :: Array{ELTYPE, 2} # [dimension, particle]
    initial_velocity    :: Array{ELTYPE, 2} # [dimension, particle]
    mass                :: Array{ELTYPE, 1} # [particle]
    pressure            :: Array{ELTYPE, 1} # [particle]
    density_calculator  :: DC
    state_equation      :: SE
    smoothing_kernel    :: K
    smoothing_length    :: ELTYPE
    viscosity           :: V
    acceleration        :: SVector{NDIMS, ELTYPE}
    cache               :: C

    # convenience constructor for passing a setup as first argument
    function FluidParticleContainer(setup, density_calculator::SummationDensity,
                                    state_equation, smoothing_kernel, smoothing_length;
                                    viscosity=NoViscosity(),
                                    acceleration=ntuple(_ -> 0.0,
                                                        size(setup.coordinates, 1)))
        return FluidParticleContainer(setup.coordinates, setup.velocities, setup.masses,
                                      density_calculator,
                                      state_equation, smoothing_kernel, smoothing_length,
                                      viscosity=viscosity, acceleration=acceleration)
    end

    # convenience constructor for passing a setup as first argument
    function FluidParticleContainer(setup, density_calculator::ContinuityDensity,
                                    state_equation, smoothing_kernel, smoothing_length;
                                    viscosity=NoViscosity(),
                                    acceleration=ntuple(_ -> 0.0,
                                                        size(setup.coordinates, 1)))
        return FluidParticleContainer(setup.coordinates, setup.velocities, setup.masses,
                                      setup.densities, density_calculator,
                                      state_equation, smoothing_kernel, smoothing_length,
                                      viscosity=viscosity, acceleration=acceleration)
    end

    function FluidParticleContainer(particle_coordinates, particle_velocities,
                                    particle_masses,
                                    density_calculator::SummationDensity, state_equation,
                                    smoothing_kernel, smoothing_length;
                                    viscosity=NoViscosity(),
                                    acceleration=ntuple(_ -> 0.0,
                                                        size(particle_coordinates, 1)))
        NDIMS = size(particle_coordinates, 1)
        ELTYPE = eltype(particle_coordinates)
        nparticles = size(particle_coordinates, 2)

        pressure = Vector{ELTYPE}(undef, nparticles)

        # Make acceleration an SVector
        acceleration_ = SVector(acceleration...)
        if length(acceleration_) != NDIMS
            throw(ArgumentError("Acceleration must be of length $NDIMS for a $(NDIMS)D problem!"))
        end

        if ndims(smoothing_kernel) != NDIMS
            throw(ArgumentError("Smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem!"))
        end

        if length(particle_masses) != nparticles
            throw(ArgumentError("'particle_masses' must be a vector of length $(n_particles)!"))
        end

        density = Vector{ELTYPE}(undef, nparticles)
        cache = (; density)

        return new{NDIMS, ELTYPE, typeof(density_calculator), typeof(state_equation),
                   typeof(smoothing_kernel), typeof(viscosity), typeof(cache)
                   }(particle_coordinates, particle_velocities, particle_masses, pressure,
                     density_calculator, state_equation, smoothing_kernel, smoothing_length,
                     viscosity, acceleration_, cache)
    end

    function FluidParticleContainer(particle_coordinates, particle_velocities,
                                    particle_masses, particle_densities,
                                    density_calculator::ContinuityDensity, state_equation,
                                    smoothing_kernel, smoothing_length;
                                    viscosity=NoViscosity(),
                                    acceleration=ntuple(_ -> 0.0,
                                                        size(particle_coordinates, 1)))
        NDIMS = size(particle_coordinates, 1)
        ELTYPE = eltype(particle_coordinates)
        nparticles = size(particle_coordinates, 2)

        pressure = Vector{ELTYPE}(undef, nparticles)

        # Make acceleration an SVector
        acceleration_ = SVector(acceleration...)
        if length(acceleration_) != NDIMS
            throw(ArgumentError("Acceleration must be of length $NDIMS for a $(NDIMS)D problem!"))
        end

        if ndims(smoothing_kernel) != NDIMS
            throw(ArgumentError("Smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem!"))
        end

        if length(particle_densities) != nparticles
            throw(ArgumentError("An initial density needs to be provided when using `ContinuityDensity`!"))
        end

        if length(particle_masses) != nparticles
            throw(ArgumentError("'particle_masses' must be a vector of length $(n_particles)!"))
        end

        initial_density = particle_densities
        cache = (; initial_density)

        return new{NDIMS, ELTYPE, typeof(density_calculator), typeof(state_equation),
                   typeof(smoothing_kernel), typeof(viscosity), typeof(cache)
                   }(particle_coordinates, particle_velocities, particle_masses, pressure,
                     density_calculator, state_equation, smoothing_kernel, smoothing_length,
                     viscosity, acceleration_, cache)
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
        summary_footer(io)
    end
end

@inline function v_nvariables(container::FluidParticleContainer)
    v_nvariables(container, container.density_calculator)
end
@inline function v_nvariables(container::FluidParticleContainer, ::SummationDensity)
    ndims(container)
end
@inline function v_nvariables(container::FluidParticleContainer, ::ContinuityDensity)
    ndims(container) + 1
end

@inline function hydrodynamic_mass(container::FluidParticleContainer, particle)
    return container.mass[particle]
end

# Nothing to initialize for this container
initialize!(container::FluidParticleContainer, neighborhood_search) = container

function update!(container::FluidParticleContainer, container_index, v, u, v_ode, u_ode,
                 semi, t)
    @unpack density_calculator = container

    compute_quantities(v, u, density_calculator, container, container_index, u_ode, semi)

    return container
end

function compute_quantities(v, u, ::ContinuityDensity, container, container_index, u_ode,
                            semi)
    compute_pressure!(container, v)
end

function compute_quantities(v, u, ::SummationDensity, container, container_index, u_ode,
                            semi)
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

    compute_pressure!(container, v)
end

# Use this function barrier and unpack inside to avoid passing closures to Polyester.jl with @batch (@threaded).
# Otherwise, @threaded does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
@inline function compute_density_per_particle(particle,
                                              u_particle_container, u_neighbor_container,
                                              particle_container::FluidParticleContainer,
                                              neighbor_container, neighborhood_search)
    @unpack cache = particle_container
    @unpack density = cache # Density is in the cache for SummationDensity

    particle_coords = current_coords(u_particle_container, particle_container, particle)
    for neighbor in eachneighbor(particle_coords, neighborhood_search)
        m_b = hydrodynamic_mass(neighbor_container, neighbor)
        neighbor_coords = current_coords(u_neighbor_container, neighbor_container,
                                         neighbor)
        distance = norm(particle_coords - neighbor_coords)

        if distance <= compact_support(particle_container)
            density[particle] += m_b * smoothing_kernel(particle_container, distance)
        end
    end
end

function compute_pressure!(container, v)
    @unpack state_equation, pressure = container

    # Note that @threaded makes this slower
    for particle in eachparticle(container)
        pressure[particle] = state_equation(particle_density(v, container, particle))
    end
end

function write_u0!(u0, container::FluidParticleContainer)
    @unpack initial_coordinates = container

    for particle in eachparticle(container)
        # Write particle coordinates
        for dim in 1:ndims(container)
            u0[dim, particle] = initial_coordinates[dim, particle]
        end
    end

    return u0
end

function write_v0!(v0, container::FluidParticleContainer)
    @unpack initial_velocity, density_calculator = container

    for particle in eachparticle(container)
        # Write particle velocities
        for dim in 1:ndims(container)
            v0[dim, particle] = initial_velocity[dim, particle]
        end
    end

    write_v0!(v0, density_calculator, container)

    return v0
end

function write_v0!(v0, ::SummationDensity, container::FluidParticleContainer)
    return v0
end

function write_v0!(v0, ::ContinuityDensity, container::FluidParticleContainer)
    @unpack cache = container
    @unpack initial_density = cache

    for particle in eachparticle(container)
        # Set particle densities
        v0[ndims(container) + 1, particle] = initial_density[particle]
    end

    return v0
end

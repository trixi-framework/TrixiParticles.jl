"""
    WeaklyCompressibleSPHSystem(particle_masses,
                                density_calculator, state_equation,
                                smoothing_kernel, smoothing_length;
                                viscosity=NoViscosity(),
                                acceleration=ntuple(_ -> 0.0, size(particle_coordinates, 1)))

System for fluid particles.
"""
struct WeaklyCompressibleSPHSystem{NDIMS, ELTYPE <: Real, DC, SE, K, V, C} <: System{NDIMS}
    mass               :: Array{ELTYPE, 1} # [particle]
    pressure           :: Array{ELTYPE, 1} # [particle]
    density_calculator :: DC
    state_equation     :: SE
    smoothing_kernel   :: K
    smoothing_length   :: ELTYPE
    viscosity          :: V
    acceleration       :: SVector{NDIMS, ELTYPE}
    cache              :: C

    function WeaklyCompressibleSPHSystem(particle_masses,
                                         density_calculator, state_equation,
                                         smoothing_kernel, smoothing_length;
                                         viscosity=NoViscosity(),
                                         acceleration=ntuple(_ -> 0.0,
                                                             ndims(smoothing_kernel)))
        NDIMS = ndims(smoothing_kernel)
        ELTYPE = eltype(particle_masses)
        n_particles = size(particle_masses, 2)

        pressure = Vector{ELTYPE}(undef, n_particles)

        # Make acceleration an SVector
        acceleration_ = SVector(acceleration...)
        if length(acceleration_) != NDIMS
            throw(ArgumentError("Acceleration must be of length $NDIMS for a $(NDIMS)D problem!"))
        end

        # TODO: Move that to the initialization when we have the initial condition
        # if ndims(smoothing_kernel) != NDIMS
        #     throw(ArgumentError("Smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem!"))
        # end
        #
        # if length(particle_masses) != n_particles
        #     throw(ArgumentError("`particle_masses` must be a vector of length $(n_particles)!"))
        # end

        cache = create_cache(n_particles, ELTYPE, density_calculator)

        return new{NDIMS, ELTYPE, typeof(density_calculator), typeof(state_equation),
                   typeof(smoothing_kernel), typeof(viscosity), typeof(cache)
                   }(particle_masses, pressure,
                     density_calculator, state_equation, smoothing_kernel, smoothing_length,
                     viscosity, acceleration_, cache)
    end
end

function create_cache(n_particles, ELTYPE, ::SummationDensity)
    density = Vector{ELTYPE}(undef, n_particles)

    return (; density)
end

function create_cache(n_particles, ELTYPE, ::ContinuityDensity)
    return (;)
end

function Base.show(io::IO, system::WeaklyCompressibleSPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "WeaklyCompressibleSPHSystem{", ndims(system), "}(")
    print(io, system.density_calculator)
    print(io, ", ", system.state_equation)
    print(io, ", ", system.smoothing_kernel)
    print(io, ", ", system.viscosity)
    print(io, ", ", system.acceleration)
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::WeaklyCompressibleSPHSystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "WeaklyCompressibleSPHSystem{$(ndims(system))}")
        summary_line(io, "#particles", nparticles(system))
        summary_line(io, "density calculator",
                     system.density_calculator |> typeof |> nameof)
        summary_line(io, "state equation", system.state_equation |> typeof |> nameof)
        summary_line(io, "smoothing kernel", system.smoothing_kernel |> typeof |> nameof)
        summary_line(io, "viscosity", system.viscosity)
        summary_line(io, "acceleration", system.acceleration)
        summary_footer(io)
    end
end

@inline function v_nvariables(system::WeaklyCompressibleSPHSystem)
    return v_nvariables(system, system.density_calculator)
end

@inline function v_nvariables(system::WeaklyCompressibleSPHSystem, density_calculator)
    return ndims(system)
end

@inline function v_nvariables(system::WeaklyCompressibleSPHSystem, ::ContinuityDensity)
    return ndims(system) + 1
end

@inline function hydrodynamic_mass(system::WeaklyCompressibleSPHSystem, particle)
    return system.mass[particle]
end

# Nothing to initialize for this system
initialize!(system::WeaklyCompressibleSPHSystem, neighborhood_search) = system

function update!(system::WeaklyCompressibleSPHSystem, system_index, v, u, v_ode,
                 u_ode,
                 semi, t)
    @unpack density_calculator = system

    compute_quantities(v, u, density_calculator, system, system_index, u_ode, semi)

    return system
end

function compute_quantities(v, u, ::ContinuityDensity, system, system_index, u_ode,
                            semi)
    compute_pressure!(system, v)
end

function compute_quantities(v, u, ::SummationDensity, system, system_index, u_ode,
                            semi)
    @unpack systems, neighborhood_searches = semi
    @unpack cache = system
    @unpack density = cache # Density is in the cache for SummationDensity

    density .= zero(eltype(density))

    # Use all other systems for the density summation
    @trixi_timeit timer() "compute density" foreach_enumerate(systems) do (neighbor_system_index,
                                                                           neighbor_system)
        u_neighbor_system = wrap_u(u_ode, neighbor_system_index,
                                   neighbor_system, semi)

        system_coords = current_coordinates(u, system)
        neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

        neighborhood_search = neighborhood_searches[system_index][neighbor_system_index]

        # Loop over all pairs of particles and neighbors within the kernel cutoff.
        for_particle_neighbor(system, neighbor_system,
                              system_coords, neighbor_coords,
                              neighborhood_search) do particle, neighbor, pos_diff, distance
            mass = hydrodynamic_mass(neighbor_system, neighbor)
            density[particle] += mass * smoothing_kernel(system, distance)
        end
    end

    compute_pressure!(system, v)
end

function compute_pressure!(system, v)
    @unpack state_equation, pressure = system

    # Note that @threaded makes this slower
    for particle in eachparticle(system)
        pressure[particle] = state_equation(particle_density(v, system, particle))
    end
end

function write_u0!(u0, system::WeaklyCompressibleSPHSystem, initial_condition)
    for particle in eachparticle(system)
        # Write particle coordinates
        for dim in 1:ndims(system)
            u0[dim, particle] = initial_condition.coordinates[dim, particle]
        end
    end

    return u0
end

function write_v0!(v0, system::WeaklyCompressibleSPHSystem, initial_condition)
    for particle in eachparticle(system)
        # Write particle velocities
        for dim in 1:ndims(system)
            v0[dim, particle] = initial_condition.velocity[dim, particle]
        end
    end

    write_v0!(v0, system.density_calculator, system, initial_condition)

    return v0
end

function write_v0!(v0, ::SummationDensity, system::WeaklyCompressibleSPHSystem,
                   initial_condition)
    return v0
end

function write_v0!(v0, ::ContinuityDensity, system::WeaklyCompressibleSPHSystem,
                   initial_condition)
    for particle in eachparticle(system)
        # Set particle densities
        v0[ndims(system) + 1, particle] = initial_condition.density[particle]
    end

    return v0
end

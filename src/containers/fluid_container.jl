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

	FluidParticleContainer(particle_coordinates, particle_velocities, particle_masses,
                           particle_densities,
						   density_calculator::ContinuityDensity, state_equation,
						   smoothing_kernel, smoothing_length, ref_density;
						   viscosity=NoViscosity(),
						   acceleration=ntuple(_ -> 0.0, size(particle_coordinates, 1)),
						   surface_tension=NoSurfaceTension(), save_forces=false)

Container for fluid particles. With [`ContinuityDensity`](@ref), the `particle_densities` array has to be passed.
"""
struct FluidParticleContainer{NDIMS, ELTYPE <: Real, DC, SE, K, V, C, SRFT, SAVE, STATE0} <:
       ParticleContainer{NDIMS}
    initial_coordinates :: Array{ELTYPE, 2} # [dimension, particle]
    initial_velocity    :: Array{ELTYPE, 2} # [dimension, particle]
    surface_normal      :: Array{ELTYPE, 2} # [dimension, particle]
    a_viscosity         :: Array{ELTYPE, 2} # [dimension, particle]
    a_surface_tension   :: Array{ELTYPE, 2} # [dimension, particle]
    a_pressure          :: Array{ELTYPE, 2} # [dimension, particle]
    mass                :: Array{ELTYPE, 1} # [particle]
    pressure            :: Array{ELTYPE, 1} # [particle]
    density_calculator  :: DC
    state_equation      :: SE
    smoothing_kernel    :: K
    smoothing_length    :: ELTYPE
    viscosity           :: V
    acceleration        :: SVector{NDIMS, ELTYPE}
    cache               :: C
    surface_tension     :: SRFT
    store_options       :: SAVE
    state0              :: STATE0

    # convenience constructor for passing a setup as first argument
    function FluidParticleContainer(setup, density_calculator::SummationDensity,
                                    state_equation, smoothing_kernel, smoothing_length, state_at_rest;
                                    viscosity=NoViscosity(),
                                    acceleration=ntuple(_ -> 0.0,
                                                        size(particle_coordinates, 1)), surface_tension=NoSurfaceTension(),
                                                        store_options=DefaultStore())
        return FluidParticleContainer(setup.coordinates, setup.velocities, setup.masses,
                                      density_calculator,
                                      state_equation, smoothing_kernel, smoothing_length, state_at_rest,
                                      viscosity=viscosity, acceleration=acceleration,
                                      surface_tension=surface_tension,
                                      store_options=store_options)
    end

    # convenience constructor for passing a setup as first argument
    function FluidParticleContainer(setup, density_calculator::ContinuityDensity,
                                    state_equation, smoothing_kernel, smoothing_length, state_at_rest;
                                    viscosity=NoViscosity(),
                                    acceleration=ntuple(_ -> 0.0,
                                                        size(particle_coordinates, 1)),
                                    surface_tension=NoSurfaceTension(),
                                    store_options=DefaultStore())
        return FluidParticleContainer(setup.coordinates, setup.velocities, setup.masses,
                                      setup.densities, density_calculator,
                                      state_equation, smoothing_kernel, smoothing_length, state_at_rest,
                                      viscosity=viscosity, acceleration=acceleration,
                                      surface_tension=surface_tension,
                                      store_options=store_options)
    end

    function FluidParticleContainer(particle_coordinates, particle_velocities,
                                    particle_masses,
                                    density_calculator::SummationDensity, state_equation,
                                    smoothing_kernel, smoothing_length, state_at_rest;
                                    viscosity=NoViscosity(),
                                    acceleration=ntuple(_ -> 0.0,
                                                        size(particle_coordinates, 1)),
                                    surface_tension=NoSurfaceTension(),
                                    store_options=DefaultStore())
        NDIMS = size(particle_coordinates, 1)
        ELTYPE = eltype(particle_masses)
        nparticles = length(particle_masses)

        if nparticles == 0
            error("particle_masses has 0-length!")
        end

        if state_at_rest === nothing
            state_at_rest = state()
        end


        pressure = Vector{ELTYPE}(undef, nparticles)

        a_surf = Array{ELTYPE, 2}(undef, NDIMS, 1)
        a_visc = Array{ELTYPE, 2}(undef, NDIMS, 1)
        a_pressure = Array{ELTYPE, 2}(undef, NDIMS, 1)
        surf_n = Array{ELTYPE, 2}(undef, NDIMS, 1)

        if surface_tension isa SurfaceTensionAkinci
            surf_n = Array{ELTYPE, 2}(undef, NDIMS, nparticles)
            println("WARNING: Result is *probably* inaccurate when used without corrections.
                     Incorrect pressure near the boundary leads the particles near walls to
                     be too far away, which leads to surface tension being applied near walls!")
        end

        if store_options isa StoreAll
            a_visc = Array{ELTYPE, 2}(undef, NDIMS, nparticles)
            a_pressure = Array{ELTYPE, 2}(undef, NDIMS, nparticles)
            if surface_tension isa AkinciTypeSurfaceTension
                a_surf = Array{ELTYPE, 2}(undef, NDIMS, nparticles)
            end
        end

        # Make acceleration an SVector
        acceleration_ = SVector(acceleration...)
        if length(acceleration_) != NDIMS
            error("Acceleration must be of length $NDIMS for a $(NDIMS)D problem")
        end

        density = Vector{ELTYPE}(undef, nparticles)
        cache = (; density)

        return new{NDIMS, ELTYPE, typeof(density_calculator), typeof(state_equation),
                   typeof(smoothing_kernel), typeof(viscosity), typeof(cache),
                   typeof(surface_tension), typeof(store_options), typeof(state_at_rest)}(particle_coordinates,
                                                                   particle_velocities,
                                                                   surf_n,
                                                                   a_visc, a_surf,
                                                                   a_pressure,
                                                                   particle_masses,
                                                                   pressure,
                                                                   density_calculator,
                                                                   state_equation,
                                                                   smoothing_kernel,
                                                                   smoothing_length,
                                                                   viscosity, acceleration_,
                                                                   cache,
                                                                   surface_tension,
                                                                   store_options,
                                                                   state_at_rest )
    end

    function FluidParticleContainer(particle_coordinates, particle_velocities,
                                    particle_masses, particle_densities,
                                    density_calculator::ContinuityDensity, state_equation,
                                    smoothing_kernel, smoothing_length, state_at_rest;
                                    viscosity=NoViscosity(),
                                    acceleration=ntuple(_ -> 0.0,
                                                        size(particle_coordinates, 1)),
                                    surface_tension=NoSurfaceTension(),
                                    store_options=DefaultStore())
        NDIMS = size(particle_coordinates, 1)
        ELTYPE = eltype(particle_masses)
        nparticles = length(particle_masses)

        if nparticles == 0
            error("particle_masses has 0-length!")
        end

        pressure = Vector{ELTYPE}(undef, nparticles)

        a_surf = Array{ELTYPE, 2}(undef, NDIMS, 1)
        a_visc = Array{ELTYPE, 2}(undef, NDIMS, 1)
        a_pressure = Array{ELTYPE, 2}(undef, NDIMS, 1)
        surf_n = Array{ELTYPE, 2}(undef, NDIMS, 1)

        if surface_tension isa SurfaceTensionAkinci
            surf_n = Array{ELTYPE, 2}(undef, NDIMS, nparticles)
            println("WARNING: Result is *probably* inaccurate when used without corrections.
                     Incorrect pressure near the boundary leads the particles near walls to
                     be too far away, which leads to surface tension being applied near walls!")
        end

        if store_options isa StoreAll
            a_visc = Array{ELTYPE, 2}(undef, NDIMS, nparticles)
            a_pressure = Array{ELTYPE, 2}(undef, NDIMS, nparticles)
            if surface_tension isa AkinciTypeSurfaceTension
                a_surf = Array{ELTYPE, 2}(undef, NDIMS, nparticles)
            end
        end

        # Make acceleration an SVector
        acceleration_ = SVector(acceleration...)
        if length(acceleration_) != NDIMS
            error("Acceleration must be of length $NDIMS for a $(NDIMS)D problem")
        end

        initial_density = particle_densities
        cache = (; initial_density)

        return new{NDIMS, ELTYPE, typeof(density_calculator), typeof(state_equation),
                   typeof(smoothing_kernel), typeof(viscosity), typeof(cache),
                   typeof(surface_tension), typeof(store_options), typeof(state_at_rest)}(particle_coordinates,
                                                                   particle_velocities,
                                                                   surf_n, a_visc, a_surf,
                                                                   a_pressure,
                                                                   particle_masses,
                                                                   pressure,
                                                                   density_calculator,
                                                                   state_equation,
                                                                   smoothing_kernel,
                                                                   smoothing_length,
                                                                   viscosity, acceleration_,
                                                                   cache,
                                                                   surface_tension,
                                                                   store_options,
                                                                   state_at_rest)
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

@inline function v_nvariables(container::FluidParticleContainer)
    v_nvariables(container, container.density_calculator)
end
@inline function v_nvariables(container::FluidParticleContainer, ::SummationDensity)
    ndims(container)
end
@inline function v_nvariables(container::FluidParticleContainer, ::ContinuityDensity)
    ndims(container) + 1
end

@inline function get_hydrodynamic_mass(particle, container::FluidParticleContainer)
    return container.mass[particle]
end

# Nothing to initialize for this container
initialize!(container::FluidParticleContainer, neighborhood_search) = container

function update!(container::FluidParticleContainer, container_index, v, u, v_ode, u_ode,
                 semi, t)
    @unpack density_calculator, surface_tension = container

    compute_quantities(v, u, density_calculator, container, container_index, u_ode, semi)

    # some surface tension models require the surface normal
    # Note: this is the most expensive step in update! when *active*!
    compute_surface_normal(surface_tension, v, u, container, container_index, u_ode,
                           v_ode, semi, t)

    return container
end

function compute_surface_normal(surface_tension::Any, v, u, container, container_index,
                                u_ode, v_ode, semi, t)
    # skip
end

function compute_surface_normal(surface_tension::SurfaceTensionAkinci, v, u, container,
                                container_index, u_ode,
                                v_ode, semi, t)
    @unpack surface_normal = container

    if t > eps() # skip depending on order boundary density is not set and will diverge
        # @efaulhaber this should be fixed in another way...
        compute_surface_normal(surface_tension, v, u, container, container_index, u_ode,
                               v_ode, semi)
    else
        # reset surface normal
        for particle in eachparticle(container)
            for i in 1:ndims(container)
                surface_normal[i, particle] = 0.0
            end
        end
    end
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
    @pixie_timeit timer() "compute density" foreach_enumerate(particle_containers) do (neighbor_container_index,
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

function compute_surface_normal(::Any, v, u, container, container_index, u_ode, v_ode, semi)
    # skip
end

function compute_surface_normal(surface_tension::SurfaceTensionAkinci, v, u, container,
                                container_index, u_ode, v_ode, semi)
    @unpack particle_containers, neighborhood_searches = semi
    @unpack surface_normal = container

    # reset surface normal
    for particle in eachparticle(container)
        for i in 1:ndims(container)
            surface_normal[i, particle] = 0.0
        end
    end

    @pixie_timeit timer() "compute surface normal" foreach_enumerate(particle_containers) do (neighbor_container_index,
                                                                                              neighbor_container)
        u_neighbor_container = wrap_u(u_ode, neighbor_container_index,
                                      neighbor_container, semi)
        v_neighbor_container = wrap_v(v_ode, neighbor_container_index,
                                      neighbor_container, semi)

        calc_normal_akinci(surface_tension, u, v_neighbor_container,
                           u_neighbor_container,
                           neighborhood_searches[container_index][neighbor_container_index],
                           container, particle_containers[neighbor_container_index])
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
    #@unpack boundary_model = neighbor_container

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

function compute_pressure!(container, v)
    @unpack state_equation, pressure = container

    # Note that @threaded makes this slower
    for particle in eachparticle(container)
        pressure[particle] = state_equation(get_particle_density(particle, v, container))
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

@inline function get_normal(particle, particle_container::FluidParticleContainer,
                            ::SurfaceTensionAkinci)
    @unpack surface_normal = particle_container
    return get_vec_field(particle, surface_normal, particle_container)
end

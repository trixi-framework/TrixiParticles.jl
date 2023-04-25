"""
    FluidParticleContainer(SPH_scheme, setup,
                           density_calculator, smoothing_kernel, smoothing_length;
                           viscosity=NoViscosity(),
                           acceleration=ntuple(_ -> 0.0, size(particle_coordinates, 1)))

    FluidParticleContainer(SPH_scheme, particle_coordinates, particle_velocities,
                           particle_masses, particle_densities,
                           density_calculator::SummationDensity,
                           smoothing_kernel, smoothing_length;
                           viscosity=NoViscosity(),
                           acceleration=ntuple(_ -> 0.0, size(particle_coordinates, 1)))

    FluidParticleContainer(SPH_scheme, particle_coordinates, particle_velocities,
                           particle_masses, particle_densities,
                           density_calculator::ContinuityDensity,
                           smoothing_kernel, smoothing_length;
                           viscosity=NoViscosity(),
                           acceleration=ntuple(_ -> 0.0, size(particle_coordinates, 1)))

Container for fluid particles.
"""
struct FluidParticleContainer{SC, NDIMS, ELTYPE <: Real, DC, K, V, C} <:
       ParticleContainer{NDIMS}
    SPH_scheme          :: SC
    initial_coordinates :: Array{ELTYPE, 2} # [dimension, particle]
    initial_velocity    :: Array{ELTYPE, 2} # [dimension, particle]
    mass                :: Array{ELTYPE, 1} # [particle]
    pressure            :: Array{ELTYPE, 1} # [particle]
    density_calculator  :: DC
    smoothing_kernel    :: K
    smoothing_length    :: ELTYPE
    viscosity           :: V
    acceleration        :: SVector{NDIMS, ELTYPE}
    cache               :: C

    # convenience constructor for passing a setup as first argument
    function FluidParticleContainer(SPH_scheme, setup, density_calculator,
                                    smoothing_kernel, smoothing_length;
                                    viscosity=NoViscosity(),
                                    acceleration=ntuple(_ -> 0.0,
                                                        size(setup.coordinates, 1)))
        return FluidParticleContainer(SPH_scheme, setup.coordinates, setup.velocities,
                                      setup.masses, setup.densities, density_calculator,
                                      smoothing_kernel, smoothing_length,
                                      viscosity=viscosity,
                                      acceleration=acceleration)
    end

    function FluidParticleContainer(SPH_scheme, particle_coordinates, particle_velocities,
                                    particle_masses, particle_densities,
                                    density_calculator, smoothing_kernel, smoothing_length;
                                    viscosity=NoViscosity(),
                                    acceleration=ntuple(_ -> 0.0,
                                                        size(particle_coordinates, 1)))
        NDIMS = size(particle_coordinates, 1)
        ELTYPE = eltype(particle_masses)
        nparticles = length(particle_masses)

        # Make acceleration an SVector
        acceleration_ = SVector(acceleration...)

        if length(acceleration_) != NDIMS
            error("Acceleration must be of length $NDIMS for a $(NDIMS)D problem")
        end

        if ndims(smoothing_kernel) != NDIMS
            error("Smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem")
        end

        pressure = Vector{ELTYPE}(undef, nparticles)

        cache = create_cache(particle_densities, density_calculator)

        return new{typeof(SPH_scheme), NDIMS, ELTYPE, typeof(density_calculator),
                   typeof(smoothing_kernel), typeof(viscosity), typeof(cache)
                   }(SPH_scheme, particle_coordinates, particle_velocities,
                     particle_masses, pressure, density_calculator, smoothing_kernel,
                     smoothing_length, viscosity, acceleration_, cache)
    end
end

function create_cache(initial_density, ::SummationDensity)
    density = similar(initial_density)

    return (; density)
end

function create_cache(initial_density, ::ContinuityDensity)
    return (; initial_density)
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

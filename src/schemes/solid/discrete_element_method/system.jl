"""
    DEMSystem(initial_condition, normal_stiffness, elastic_modulus, poissons_ratio;
     damping_coefficient=0.0001, acceleration=ntuple(_ -> 0.0, NDIMS), source_terms=nothing)

Constructs a Discrete Element Method (DEM) system for numerically simulating the dynamics of
granular and particulate matter. DEM is employed to simulate and analyze the motion,
interactions, and collective behavior of assemblies of discrete, solid particles, typically
under mechanical loading. The model accounts for individual particle characteristics
and implements interaction laws that govern contact forces (normal and tangential), based on
specified material properties and contact mechanics.

# Arguments
 - `initial_condition`: Initial condition of the system, encapsulating the initial positions,
    velocities, masses, and radii of particles.
 - `normal_stiffness`: Normal stiffness coefficient for particle-particle and particle-wall contacts.
 - `elastic_modulus`: Elastic modulus for this particle system.
 - `poissons_ratio`: Poisson ratio for this particle system.

# Keywords
 - `acceleration`: Global acceleration vector applied to the system, such as gravity. Specified as
    an `SVector` of length `NDIMS`, with a default of zero in each dimension.
 - `source_terms`: Optional; additional forces or modifications to particle dynamics not
    captured by standard DEM interactions, such as electromagnetic forces or user-defined perturbations.
 - `damping_coefficient=0.0001`: Set a damping coefficient for the collision interactions.

 !!! warning "Experimental Implementation"
    This is an experimental feature and may change in a future releases.
"""
struct DEMSystem{NDIMS, ELTYPE <: Real, ARRAY1D, ST} <: SolidSystem{NDIMS}
    initial_condition   :: InitialCondition{ELTYPE}
    mass                :: ARRAY1D               # [particle]
    radius              :: ARRAY1D               # [particle]
    elastic_modulus     :: ELTYPE
    poissons_ratio      :: ELTYPE
    normal_stiffness    :: ELTYPE
    damping_coefficient :: ELTYPE
    acceleration        :: SVector{NDIMS, ELTYPE}
    source_terms        :: ST

    function DEMSystem(initial_condition, normal_stiffness, elastic_modulus, poissons_ratio;
                       damping_coefficient=0.0001,
                       acceleration=ntuple(_ -> 0.0,
                                           ndims(initial_condition)), source_terms=nothing)
        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)

        mass = copy(initial_condition.mass)
        radius = 0.5 * initial_condition.particle_spacing * ones(length(mass))

        # Make acceleration an SVector
        acceleration_ = SVector(acceleration...)
        if length(acceleration_) != NDIMS
            throw(ArgumentError("`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"))
        end

        return new{NDIMS, ELTYPE, typeof(mass),
                   typeof(source_terms)}(initial_condition, mass, radius, elastic_modulus,
                                         poissons_ratio, normal_stiffness,
                                         damping_coefficient, acceleration_, source_terms)
    end
end

function Base.show(io::IO, system::DEMSystem)
    @nospecialize system # reduce precompilation time

    print(io, "DEMSystem{", ndims(system), "}(")
    print(io, system.initial_condition)
    print(io, ", ", system.elastic_modulus)
    print(io, ", ", system.poissons_ratio)
    print(io, ", ", system.normal_stiffness)
    print(io, ", ", system.damping_coefficient)
    print(io, ") with ", TrixiParticles.nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::DEMSystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        TrixiParticles.summary_header(io, "DEMSystem{$(ndims(system))}")
        TrixiParticles.summary_line(io, "#particles", TrixiParticles.nparticles(system))
        TrixiParticles.summary_line(io, "elastic_modulus", system.elastic_modulus)
        TrixiParticles.summary_line(io, "poissons_ratio", system.poissons_ratio)
        TrixiParticles.summary_line(io, "normal_stiffness", system.normal_stiffness)
        TrixiParticles.summary_line(io, "damping_coefficient", system.damping_coefficient)
        TrixiParticles.summary_footer(io)
    end
end

timer_name(::DEMSystem) = "solid"

function TrixiParticles.write_u0!(u0, system::DEMSystem)
    u0 .= system.initial_condition.coordinates
    return u0
end

function TrixiParticles.write_v0!(v0, system::DEMSystem)
    v0 .= system.initial_condition.velocity
    return v0
end

# Nothing to initialize for this system
initialize!(system::DEMSystem, neighborhood_search) = system

function compact_support(system::DEMSystem, neighbor::DEMSystem)
    # we for now assume that the compact support is 3 * radius
    # todo: needs to be changed for more complex simulations
    return 3 * max(maximum(system.radius), maximum(neighbor.radius))
end

function compact_support(system::DEMSystem, neighbor)
    # we for now assume that the compact support is 3 * radius
    # todo: needs to be changed for more complex simulations
    return 3 * maximum(system.radius)
end

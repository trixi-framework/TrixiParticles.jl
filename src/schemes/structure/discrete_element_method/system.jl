"""
    DEMSystem(initial_condition, contact_model; damping_coefficient=0.0001,
              acceleration=ntuple(_ -> 0.0, ndims(initial_condition)), source_terms=nothing,
              radius=nothing)

Constructs a Discrete Element Method (DEM) system for numerically simulating the dynamics of
granular and particulate matter. DEM is employed to simulate and analyze the motion,
interactions, and collective behavior of assemblies of discrete, solid particles, typically
under mechanical loading. The model accounts for individual particle characteristics
and implements interaction laws that govern contact forces (normal and tangential), based on
specified material properties and contact mechanics.

# Arguments
 - `initial_condition`: Initial condition of the system, encapsulating the initial positions,
    velocities, masses, and radii of particles.
 - `contact_model`: Contact model used for particle interactions.

# Keywords
 - `acceleration`: Global acceleration vector applied to the system, such as gravity. Specified as
    an `SVector` of length `NDIMS`, with a default of zero in each dimension.
 - `source_terms`: Optional; additional forces or modifications to particle dynamics not
    captured by standard DEM interactions, such as electromagnetic forces or user-defined perturbations.
 - `damping_coefficient=0.0001`: Set a damping coefficient for the collision interactions.
 - `radius=nothing`: Specifies the radius of the particles, defaults to `initial_condition.particle_spacing / 2`.

!!! warning "Experimental Implementation"
    This is an experimental feature and may change in a future releases.

## References
[Bicanic2004](@cite), [Cundall1979](@cite), [DiRenzo2004](@cite)
"""
struct DEMSystem{NDIMS, ELTYPE <: Real, IC, ARRAY1D, ST,
                 CM} <: AbstractStructureSystem{NDIMS}
    initial_condition   :: IC
    mass                :: ARRAY1D               # [particle]
    radius              :: ARRAY1D               # [particle]
    damping_coefficient :: ELTYPE
    acceleration        :: SVector{NDIMS, ELTYPE}
    source_terms        :: ST
    contact_model       :: CM
end

# The default constructor needs to be accessible for Adapt.jl to work with this struct.
# See the comments in general/gpu.jl for more details.
function DEMSystem(initial_condition, contact_model; damping_coefficient=0.0001,
                   acceleration=ntuple(_ -> 0.0,
                                       ndims(initial_condition)), source_terms=nothing,
                   radius=nothing)
    NDIMS = ndims(initial_condition)
    ELTYPE = eltype(initial_condition)

    mass = copy(initial_condition.mass)

    if isnothing(radius)
        radius = initial_condition.particle_spacing * ones(ELTYPE, length(mass)) / 2
    else
        mass = (radius / (initial_condition.particle_spacing / 2))^3 * mass
        radius = radius * ones(ELTYPE, length(mass))
    end

    # Make acceleration an SVector
    acceleration_ = SVector(acceleration...)
    if length(acceleration_) != NDIMS
        throw(ArgumentError("`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"))
    end

    return DEMSystem(initial_condition, mass, radius,
                     damping_coefficient, acceleration_, source_terms,
                     contact_model)
end

function Base.show(io::IO, system::DEMSystem)
    @nospecialize system # reduce precompilation time

    print(io, "DEMSystem{", ndims(system), "}(")
    print(io, system.initial_condition, ", ")
    # TODO: Dispatch on the type of the contact_model to show the relevant parameters.
    if system.contact_model isa HertzContactModel
        print(io, "HertzContactModel: elastic_modulus = ",
              system.contact_model.elastic_modulus, ", poissons_ratio = ",
              system.contact_model.poissons_ratio)
    elseif system.contact_model isa LinearContactModel
        print(io, "LinearContactModel: normal_stiffness = ",
              system.contact_model.normal_stiffness)
    else
        print(io, "UnknownContactModel")
    end
    print(io, ", damping_coefficient = ", system.damping_coefficient, ")")
    print(io, " with ", TrixiParticles.nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::DEMSystem)
    @nospecialize system
    if get(io, :compact, false)
        show(io, system)
    else
        TrixiParticles.summary_header(io, "DEMSystem{$(ndims(system))}")
        TrixiParticles.summary_line(io, "#particles", TrixiParticles.nparticles(system))
        # Display contact model specific parameters.
        if system.contact_model isa HertzContactModel
            TrixiParticles.summary_line(io, "elastic_modulus",
                                        system.contact_model.elastic_modulus)
            TrixiParticles.summary_line(io, "poissons_ratio",
                                        system.contact_model.poissons_ratio)
        elseif system.contact_model isa LinearContactModel
            TrixiParticles.summary_line(io, "normal_stiffness",
                                        system.contact_model.normal_stiffness)
        end
        TrixiParticles.summary_line(io, "damping_coefficient", system.damping_coefficient)
        TrixiParticles.summary_footer(io)
    end
end

@inline function Base.eltype(::DEMSystem{<:Any, ELTYPE}) where {ELTYPE}
    return ELTYPE
end

timer_name(::DEMSystem) = "solid"

function TrixiParticles.write_u0!(u0, system::DEMSystem)
    copyto!(u0, system.initial_condition.coordinates)
    return u0
end

function TrixiParticles.write_v0!(v0, system::DEMSystem)
    copyto!(v0, system.initial_condition.velocity)
    return v0
end

# Nothing to initialize for this system
initialize!(system::DEMSystem, semi) = system

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

@inline function hydrodynamic_mass(system::DEMSystem, particle)
    return system.mass[particle]
end

@inline function particle_radius(system::DEMSystem, particle)
    return system.radius[particle]
end

function system_data(system::DEMSystem, dv_ode, du_ode, v_ode, u_ode, semi)
    (; mass, radius, damping_coefficient) = system

    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    coordinates = current_coordinates(u, system)
    velocity = current_velocity(v, system)

    return (; coordinates, velocity, mass, radius, damping_coefficient)
end

function available_data(::DEMSystem)
    return (:coordinates, :velocity, :mass, :radius, :damping_coefficient)
end

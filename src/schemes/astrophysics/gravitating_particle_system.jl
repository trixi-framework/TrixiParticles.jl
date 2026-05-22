@doc raw"""
    GravitatingParticleSystem(initial_condition;
                              acceleration=ntuple(_ -> zero(eltype(initial_condition)),
                                                   ndims(initial_condition)),
                              particle_ids=axes(initial_condition.coordinates, 2),
                              gravity=NewtonianGravity())
    GravitatingParticleSystem(; coordinates, mass, velocity=nothing,
                              acceleration=nothing,
                              particle_ids=axes(coordinates, 2),
                              gravity=NewtonianGravity())

Point-mass particle system for gravity-only simulations.

`coordinates` is an `NDIMS x nparticles` matrix with `NDIMS == 2` or `NDIMS == 3`.
The system stores mass, position, velocity, acceleration, and particle IDs. It has
no SPH density, pressure, smoothing length, or hydrodynamic state.

The keyword-only constructor is a convenience wrapper around an [`InitialCondition`](@ref)
with explicit particle masses.

The `acceleration` field is an optional global external acceleration. Pairwise
self-gravity is controlled by `gravity`; by default, `NewtonianGravity()` uses the
unitless convention ``G = 1``.
"""
struct GravitatingParticleSystem{NDIMS, ELTYPE <: Real, IC, MASS, IDS,
                                 GR} <: AbstractAstrophysicsSystem{NDIMS}
    initial_condition :: IC
    mass              :: MASS
    acceleration      :: SVector{NDIMS, ELTYPE}
    particle_ids      :: IDS
    gravity           :: GR
    buffer            :: Nothing
end

function GravitatingParticleSystem(initial_condition::InitialCondition;
                                   acceleration=ntuple(_ -> zero(eltype(initial_condition)),
                                                       ndims(initial_condition)),
                                   particle_ids=axes(initial_condition.coordinates, 2),
                                   gravity=NewtonianGravity())
    NDIMS = ndims(initial_condition)
    NDIMS in (2, 3) ||
        throw(ArgumentError("`initial_condition` must contain 2D or 3D positions"))

    ELTYPE = eltype(initial_condition)
    n_particles = nparticles(initial_condition)

    acceleration_ = SVector(acceleration...)
    length(acceleration_) == NDIMS ||
        throw(ArgumentError("`acceleration` must be of length $NDIMS for " *
                            "a $(NDIMS)D problem"))
    acceleration__ = convert(SVector{NDIMS, ELTYPE}, acceleration_)

    particle_ids_ = collect(particle_ids)
    length(particle_ids_) == n_particles ||
        throw(ArgumentError("Expected `length(particle_ids) == " *
                            "nparticles(initial_condition)`"))
    eltype(particle_ids_) <: Integer ||
        throw(ArgumentError("`particle_ids` must contain integers"))
    allunique(particle_ids_) ||
        throw(ArgumentError("`particle_ids` must be unique"))

    gravity === nothing || gravity isa AbstractGravityModel ||
        throw(ArgumentError("`gravity` must be `nothing` or a gravity model"))
    gravity_ = convert_gravity_model(gravity, ELTYPE)

    all(isfinite, initial_condition.coordinates) ||
        throw(ArgumentError("`coordinates` must be finite"))
    all(isfinite, initial_condition.velocity) ||
        throw(ArgumentError("`velocity` must be finite"))
    all(isfinite, acceleration__) ||
        throw(ArgumentError("`acceleration` must be finite"))

    mass = copy(initial_condition.mass)
    all(mass -> isfinite(mass) && mass >= zero(mass), mass) ||
        throw(ArgumentError("`mass` must be non-negative and finite"))

    return GravitatingParticleSystem{NDIMS, ELTYPE, typeof(initial_condition),
                                     typeof(mass), typeof(particle_ids_),
                                     typeof(gravity_)}(initial_condition, mass,
                                                       acceleration__,
                                                       particle_ids_, gravity_,
                                                       nothing)
end

function GravitatingParticleSystem(; coordinates, mass, velocity=nothing,
                                   acceleration=nothing,
                                   particle_ids=axes(coordinates, 2),
                                   gravity=NewtonianGravity())
    coordinates isa AbstractMatrix ||
        throw(ArgumentError("`coordinates` must be a matrix"))

    NDIMS = size(coordinates, 1)
    NDIMS in (2, 3) ||
        throw(ArgumentError("`coordinates` must contain 2D or 3D positions"))

    n_particles = size(coordinates, 2)
    length(mass) == n_particles ||
        throw(ArgumentError("Expected `length(mass) == size(coordinates, 2)`"))

    velocity_ = isnothing(velocity) ?
                zeros(eltype(coordinates), NDIMS, n_particles) : velocity

    size(velocity_) == size(coordinates) ||
        throw(ArgumentError("`velocity` must have the same size as `coordinates`"))

    ELTYPE = floating_point_type(promote_type(eltype(coordinates), eltype(velocity_),
                                              eltype(mass)))

    initial_condition = InitialCondition(; coordinates=Array{ELTYPE}(coordinates),
                                         velocity=Array{ELTYPE}(velocity_),
                                         mass=Array{ELTYPE}(mass),
                                         density=ones(ELTYPE, n_particles),
                                         particle_spacing=-one(ELTYPE))

    if isnothing(acceleration)
        return GravitatingParticleSystem(initial_condition; particle_ids, gravity)
    end

    return GravitatingParticleSystem(initial_condition; acceleration, particle_ids, gravity)
end

function Base.show(io::IO, system::GravitatingParticleSystem)
    @nospecialize system

    print(io, "GravitatingParticleSystem{", ndims(system), "}(")
    print(io, system.initial_condition)
    print(io, ", ")
    print(io, system.gravity)
    print(io, ", ", system.acceleration)
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::GravitatingParticleSystem)
    @nospecialize system

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "GravitatingParticleSystem{$(ndims(system))}")
        summary_line(io, "#particles", nparticles(system))
        summary_line(io, "gravity", nameof(typeof(system.gravity)))
        summary_line(io, "acceleration", system.acceleration)
        summary_line(io, "eltype", "$(eltype(system))")
        summary_line(io, "coordinate eltype", "$(coordinates_eltype(system))")
        summary_footer(io)
    end
end

@inline function floating_point_type(::Type{ELTYPE}) where {ELTYPE}
    return typeof(float(zero(ELTYPE)))
end

@inline convert_gravity_model(::Nothing, ::Type{ELTYPE}) where {ELTYPE} = nothing

@inline convert_gravity_model(gravity::AbstractGravityModel, ::Type) = gravity

@inline function convert_gravity_model(gravity::NewtonianGravity,
                                       ::Type{ELTYPE}) where {ELTYPE}
    return NewtonianGravity(;
                            gravitational_constant=convert(ELTYPE,
                                                           gravity.gravitational_constant),
                            softening=copy_softening_model(gravity.softening, ELTYPE),
                            cutoff_radius=convert(ELTYPE, gravity.cutoff_radius))
end

@inline function Base.eltype(::GravitatingParticleSystem{NDIMS, ELTYPE}) where {NDIMS,
                                                                                ELTYPE}
    return ELTYPE
end

timer_name(::GravitatingParticleSystem) = "gravity"

@inline function initial_coordinates(system::GravitatingParticleSystem)
    return system.initial_condition.coordinates
end

function write_u0!(u0, system::GravitatingParticleSystem)
    u0 .= system.initial_condition.coordinates

    return u0
end

function write_v0!(v0, system::GravitatingParticleSystem)
    v0 .= system.initial_condition.velocity

    return v0
end

@inline function gravitational_mass(system::GravitatingParticleSystem, particle)
    return system.mass[particle]
end

@inline function current_acceleration(system::GravitatingParticleSystem, particle)
    return system.acceleration
end

@inline gravity_model(system::GravitatingParticleSystem) = system.gravity

@inline function compact_support(system::GravitatingParticleSystem,
                                 neighbor::GravitatingParticleSystem)
    gravity = gravity_model(system, neighbor)
    gravity === nothing && return zero(eltype(system))

    return convert(eltype(system), gravity_cutoff_radius(gravity))
end

function update_nhs!(neighborhood_search,
                     system::GravitatingParticleSystem,
                     neighbor::GravitatingParticleSystem,
                     u_system, u_neighbor, semi)
    update!(neighborhood_search,
            current_position(u_system, system),
            current_position(u_neighbor, neighbor),
            semi, points_moving=(true, true),
            eachindex_y=each_active_particle(neighbor))
end

function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::GravitatingParticleSystem,
                   neighbor_system::GravitatingParticleSystem, semi)
    gravity = gravity_model(particle_system, neighbor_system)
    gravity === nothing && return dv

    system_coords = current_position(u_particle_system, particle_system)
    neighbor_coords = current_position(u_neighbor_system, neighbor_system)

    points = each_integrated_particle(particle_system)
    foreach_point_neighbor(particle_system, neighbor_system,
                           system_coords, neighbor_coords, semi;
                           points=points) do particle, neighbor, pos_diff, distance
        particle_system === neighbor_system && particle === neighbor && return
        gravity_acceleration!(dv, gravity, particle_system, neighbor_system,
                              particle, neighbor, pos_diff, distance)
    end

    return dv
end

function finalize_interaction!(system::GravitatingParticleSystem, dv, v, u,
                               dv_ode, v_ode, u_ode, semi)
    @threaded semi for particle in each_integrated_particle(system)
        acceleration = current_acceleration(system, particle)

        @inbounds for i in 1:ndims(system)
            dv[i, particle] += acceleration[i]
        end
    end

    return system
end

@inline function particle_spacing(system::GravitatingParticleSystem, particle)
    return system.initial_condition.particle_spacing
end

vtkname(system::GravitatingParticleSystem) = "gravitating_particles"

function write2vtk!(vtk, v, u, t, system::GravitatingParticleSystem)
    vtk["velocity"] = view(v, 1:ndims(system), :)
    vtk["mass"] = system.mass
    vtk["external_acceleration"] = [system.acceleration for _ in eachparticle(system)]
    vtk["particle_id"] = system.particle_ids

    return vtk
end

function system_data(system::GravitatingParticleSystem, dv_ode, du_ode,
                     v_ode, u_ode, semi)
    dv = wrap_v(dv_ode, system, semi)
    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    return (; coordinates=current_position(u, system),
            velocity=current_velocity(v, system),
            mass=system.mass,
            acceleration=dv,
            external_acceleration=system.acceleration,
            particle_id=system.particle_ids)
end

function available_data(::GravitatingParticleSystem)
    return (:coordinates, :velocity, :mass, :acceleration, :external_acceleration,
            :particle_id)
end

function add_system_data!(system_data, system::GravitatingParticleSystem)
    system_data["system_type"] = type2string(system)
    system_data["external_acceleration"] = system.acceleration
    if system.gravity !== nothing
        system_data["gravity_model"] = type2string(system.gravity)
    end
    if system.gravity isa NewtonianGravity
        system_data["gravitational_constant"] = system.gravity.gravitational_constant
    end

    return system_data
end

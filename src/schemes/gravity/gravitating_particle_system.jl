@doc raw"""
    GravitatingParticleSystem(; coordinates, mass, velocity=nothing,
                              acceleration=nothing,
                              particle_ids=1:nparticles,
                              gravity=NewtonianGravity())

Point-mass particle system for gravity-only simulations.

`coordinates` is an `NDIMS x nparticles` matrix with `NDIMS == 2` or `NDIMS == 3`.
The system stores mass, position, velocity, acceleration, and particle IDs. It has
no SPH density, pressure, smoothing length, or hydrodynamic state.

The `acceleration` field is an optional per-particle external acceleration. Pairwise
self-gravity is controlled by `gravity`; by default `NewtonianGravity()` uses the
unitless convention ``G = 1``.
"""
struct GravitatingParticleSystem{NDIMS, ELTYPE <: Real, COORDS, VELOCITY, MASS,
                                 ACCELERATION, IDS, GR} <: AbstractSystem{NDIMS}
    coordinates  :: COORDS
    velocity     :: VELOCITY
    mass         :: MASS
    acceleration :: ACCELERATION
    particle_ids :: IDS
    gravity      :: GR
    buffer       :: Nothing

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
        acceleration_ = isnothing(acceleration) ?
                        zeros(eltype(coordinates), NDIMS, n_particles) : acceleration

        size(velocity_) == size(coordinates) ||
            throw(ArgumentError("`velocity` must have the same size as `coordinates`"))
        size(acceleration_) == size(coordinates) ||
            throw(ArgumentError("`acceleration` must have the same size as `coordinates`"))

        particle_ids_ = collect(particle_ids)
        length(particle_ids_) == n_particles ||
            throw(ArgumentError("Expected `length(particle_ids) == size(coordinates, 2)`"))
        eltype(particle_ids_) <: Integer ||
            throw(ArgumentError("`particle_ids` must contain integers"))
        allunique(particle_ids_) ||
            throw(ArgumentError("`particle_ids` must be unique"))

        gravity === nothing || gravity isa NewtonianGravity ||
            throw(ArgumentError("`gravity` must be `nothing` or a `NewtonianGravity` model"))

        ELTYPE = floating_point_type(promote_type(eltype(coordinates), eltype(velocity_),
                                                  eltype(mass), eltype(acceleration_)))

        coordinates__ = Array{ELTYPE}(coordinates)
        velocity__ = Array{ELTYPE}(velocity_)
        mass__ = Array{ELTYPE}(mass)
        acceleration__ = Array{ELTYPE}(acceleration_)

        all(isfinite, coordinates__) ||
            throw(ArgumentError("`coordinates` must be finite"))
        all(isfinite, velocity__) ||
            throw(ArgumentError("`velocity` must be finite"))
        all(isfinite, acceleration__) ||
            throw(ArgumentError("`acceleration` must be finite"))
        all(mass -> isfinite(mass) && mass >= zero(mass), mass__) ||
            throw(ArgumentError("`mass` must be non-negative and finite"))

        gravity_ = convert_gravity_model(gravity, ELTYPE)

        return new{NDIMS, ELTYPE, typeof(coordinates__), typeof(velocity__),
                   typeof(mass__), typeof(acceleration__), typeof(particle_ids_),
                   typeof(gravity_)}(coordinates__, velocity__, mass__, acceleration__,
                                     particle_ids_, gravity_, nothing)
    end
end

@inline function floating_point_type(::Type{ELTYPE}) where {ELTYPE}
    return typeof(float(zero(ELTYPE)))
end

@inline convert_gravity_model(::Nothing, ::Type{ELTYPE}) where {ELTYPE} = nothing

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

@inline initial_coordinates(system::GravitatingParticleSystem) = system.coordinates

function write_u0!(u0, system::GravitatingParticleSystem)
    u0 .= system.coordinates

    return u0
end

function write_v0!(v0, system::GravitatingParticleSystem)
    v0 .= system.velocity

    return v0
end

@inline gravitational_mass(system::GravitatingParticleSystem, particle) = system.mass[particle]

@propagate_inbounds function current_acceleration(system::GravitatingParticleSystem,
                                                  particle)
    return extract_svector(system.acceleration, system, particle)
end

@inline gravity_model(system::GravitatingParticleSystem) = system.gravity

@inline function compact_support(system::GravitatingParticleSystem,
                                 neighbor::GravitatingParticleSystem)
    gravity = gravity_model(system, neighbor)
    gravity === nothing && return zero(eltype(system))

    return gravity.cutoff_radius
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
    return -one(eltype(system))
end

vtkname(system::GravitatingParticleSystem) = "gravitating_particles"

function write2vtk!(vtk, v, u, t, system::GravitatingParticleSystem)
    vtk["velocity"] = view(v, 1:ndims(system), :)
    vtk["mass"] = system.mass
    vtk["acceleration"] = system.acceleration
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
            particle_id=system.particle_ids)
end

function available_data(::GravitatingParticleSystem)
    return (:coordinates, :velocity, :mass, :acceleration, :particle_id)
end

function add_system_data!(system_data, system::GravitatingParticleSystem)
    system_data["system_type"] = type2string(system)
    system_data["acceleration"] = system.acceleration
    if system.gravity !== nothing
        system_data["gravitational_constant"] = system.gravity.gravitational_constant
        system_data["gravity_model"] = type2string(system.gravity)
    end

    return system_data
end

function Base.show(io::IO, system::GravitatingParticleSystem)
    print(io, "GravitatingParticleSystem{", ndims(system), "}() with ")
    print(io, nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::GravitatingParticleSystem)
    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "GravitatingParticleSystem{$(ndims(system))}")
        summary_line(io, "#particles", nparticles(system))
        summary_line(io, "gravity", nameof(typeof(system.gravity)))
        summary_footer(io)
    end
end

using TrixiParticles
using LinearAlgebra

struct NBodySystem{NDIMS, ELTYPE <: Real, IC, GR, FAST} <:
       TrixiParticles.AbstractSystem{NDIMS}
    initial_condition :: IC
    mass              :: Array{ELTYPE, 1} # [particle]
    G                 :: ELTYPE
    gravity           :: GR
    buffer            :: Nothing

    function NBodySystem(initial_condition, gravity::NewtonianGravity)
        mass = copy(initial_condition.mass)
        gravitational_constant = convert(eltype(mass), gravity.gravitational_constant)

        new{size(initial_condition.coordinates, 1),
            eltype(mass), typeof(initial_condition), typeof(gravity),
            iszero(gravity.softening_length) && isinf(gravity.cutoff_radius)}(initial_condition,
                                                                              mass,
                                                                              gravitational_constant,
                                                                              gravity,
                                                                              nothing)
    end
end

function NBodySystem(initial_condition, gravitational_constant)
    gravity = NewtonianGravity(; gravitational_constant)

    return NBodySystem(initial_condition, gravity)
end

TrixiParticles.timer_name(::NBodySystem) = "nbody"

@inline Base.eltype(system::NBodySystem{NDIMS, ELTYPE}) where {NDIMS, ELTYPE} = ELTYPE

@inline TrixiParticles.gravity_model(system::NBodySystem) = system.gravity

function TrixiParticles.write_u0!(u0, system::NBodySystem)
    u0 .= system.initial_condition.coordinates

    return u0
end

function TrixiParticles.write_v0!(v0, system::NBodySystem)
    v0 .= system.initial_condition.velocity

    return v0
end

# NHS update
function TrixiParticles.update_nhs!(neighborhood_search,
                                    system::NBodySystem, neighbor::NBodySystem,
                                    u_system, u_neighbor, semi)
    TrixiParticles.PointNeighbors.update!(neighborhood_search,
                                          u_system, u_neighbor,
                                          points_moving=(true, true))
end

function TrixiParticles.compact_support(system::NBodySystem,
                                        neighbor::NBodySystem)
    return TrixiParticles.gravity_model(system, neighbor).cutoff_radius
end

@inline function TrixiParticles.interact!(dv, v_particle_system, u_particle_system,
                                          v_neighbor_system, u_neighbor_system,
                                          particle_system::NBodySystem{NDIMS, ELTYPE, IC,
                                                                       GR, true},
                                          neighbor_system::NBodySystem,
                                          semi) where {NDIMS, ELTYPE, IC, GR}
    (; mass, G) = neighbor_system

    system_coords = TrixiParticles.current_coordinates(u_particle_system, particle_system)
    neighbor_coords = TrixiParticles.current_coordinates(u_neighbor_system, neighbor_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    TrixiParticles.foreach_point_neighbor(particle_system, neighbor_system,
                                          system_coords, neighbor_coords,
                                          semi) do particle, neighbor, pos_diff, distance
        # No interaction of a particle with itself
        particle_system === neighbor_system && particle === neighbor && return

        tmp = -G * mass[neighbor] * (1 / distance^3)

        @inbounds for i in 1:ndims(particle_system)
            dv[i, particle] += tmp * pos_diff[i]
        end
    end

    return dv
end

@inline function TrixiParticles.interact!(dv, v_particle_system, u_particle_system,
                                          v_neighbor_system, u_neighbor_system,
                                          particle_system::NBodySystem{NDIMS, ELTYPE, IC,
                                                                       GR, false},
                                          neighbor_system::NBodySystem,
                                          semi) where {NDIMS, ELTYPE, IC, GR}
    (; mass) = neighbor_system
    gravity = TrixiParticles.gravity_model(particle_system, neighbor_system)

    system_coords = TrixiParticles.current_coordinates(u_particle_system, particle_system)
    neighbor_coords = TrixiParticles.current_coordinates(u_neighbor_system, neighbor_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    TrixiParticles.foreach_point_neighbor(particle_system, neighbor_system,
                                          system_coords, neighbor_coords,
                                          semi) do particle, neighbor, pos_diff, distance
        # No interaction of a particle with itself
        particle_system === neighbor_system && particle === neighbor && return

        dv_gravity = TrixiParticles.gravity_acceleration(gravity, pos_diff, distance,
                                                         mass[neighbor])

        @inbounds for i in 1:ndims(particle_system)
            dv[i, particle] += dv_gravity[i]
        end
    end

    return dv
end

function energy(v_ode, u_ode, system, semi)
    (; mass) = system
    gravity = TrixiParticles.gravity_model(system)
    (; gravitational_constant, softening_length, cutoff_radius) = gravity

    e = zero(eltype(system))

    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)

    for particle in TrixiParticles.eachparticle(system)
        e += 0.5 * mass[particle] *
             sum(TrixiParticles.current_velocity(v, system, particle) .^ 2)

        particle_coords = TrixiParticles.current_coords(u, system, particle)
        for neighbor in (particle + 1):TrixiParticles.nparticles(system)
            neighbor_coords = TrixiParticles.current_coords(u, system, neighbor)

            pos_diff = particle_coords - neighbor_coords
            distance = norm(pos_diff)

            if distance <= cutoff_radius
                softened_distance = sqrt(distance^2 + softening_length^2)
                e -= gravitational_constant * mass[particle] * mass[neighbor] /
                     softened_distance
            end
        end
    end

    return e
end

TrixiParticles.vtkname(system::NBodySystem) = "n-body"

function TrixiParticles.write2vtk!(vtk, v, u, t, system::NBodySystem)
    (; mass) = system

    vtk["velocity"] = v
    vtk["mass"] = mass

    return vtk
end

function TrixiParticles.add_system_data!(system_data, system::NBodySystem)
    return system_data
end

function Base.show(io::IO, system::NBodySystem)
    print(io, "NBodySystem{", ndims(system), "}() with ")
    print(io, TrixiParticles.nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::NBodySystem)
    if get(io, :compact, false)
        show(io, system)
    else
        TrixiParticles.summary_header(io, "NBodySystem{$(ndims(system))}")
        TrixiParticles.summary_line(io, "#particles", TrixiParticles.nparticles(system))
        TrixiParticles.summary_footer(io)
    end
end

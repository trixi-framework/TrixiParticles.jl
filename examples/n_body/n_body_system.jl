using TrixiParticles
using LinearAlgebra

struct NBodySystem{NDIMS, ELTYPE <: Real, IC} <: TrixiParticles.AbstractSystem{NDIMS}
    initial_condition :: IC
    mass              :: Array{ELTYPE, 1} # [particle]
    G                 :: ELTYPE
    buffer            :: Nothing

    function NBodySystem(initial_condition, G)
        mass = copy(initial_condition.mass)

        new{size(initial_condition.coordinates, 1),
            eltype(mass), typeof(initial_condition)}(initial_condition, mass, G, nothing)
    end
end

TrixiParticles.timer_name(::NBodySystem) = "nbody"

@inline Base.eltype(system::NBodySystem{NDIMS, ELTYPE}) where {NDIMS, ELTYPE} = ELTYPE

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
    # There is no cutoff. All particles interact with each other.
    return Inf
end

function TrixiParticles.interact!(dv, v_particle_system, u_particle_system,
                                  v_neighbor_system, u_neighbor_system,
                                  particle_system::NBodySystem,
                                  neighbor_system::NBodySystem, semi)
    (; mass, G) = neighbor_system

    system_coords = TrixiParticles.current_coordinates(u_particle_system, particle_system)
    neighbor_coords = TrixiParticles.current_coordinates(u_neighbor_system, neighbor_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    TrixiParticles.foreach_point_neighbor(particle_system, neighbor_system,
                                          system_coords, neighbor_coords,
                                          semi) do particle, neighbor, pos_diff, distance
        # No interaction of a particle with itself
        particle_system === neighbor_system && particle === neighbor && return

        # Original version
        # dv = -G * mass[neighbor] * pos_diff / norm(pos_diff)^3

        # Multiplying by pos_diff later makes this slightly faster
        # Multiplying by (1 / norm) is also faster than dividing by norm
        tmp = -G * mass[neighbor] * (1 / distance^3)

        @inbounds for i in 1:ndims(particle_system)
            dv[i, particle] += tmp * pos_diff[i]
        end
    end

    return dv
end

function energy(v_ode, u_ode, system, semi)
    (; mass) = system

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

            e -= mass[particle] * mass[neighbor] / distance
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

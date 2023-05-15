using TrixiParticles
using LinearAlgebra

struct NBodyContainer{NDIMS, ELTYPE <: Real} <: TrixiParticles.ParticleContainer{NDIMS}
    initial_coordinates :: Array{ELTYPE, 2} # [dimension, particle]
    initial_velocity    :: Array{ELTYPE, 2} # [dimension, particle]
    mass                :: Array{ELTYPE, 1} # [particle]
    G                   :: ELTYPE

    function NBodyContainer(coordinates, velocities, masses, G)
        new{size(coordinates, 1), eltype(coordinates)}(coordinates, velocities, masses, G)
    end
end

@inline function TrixiParticles.add_acceleration!(dv, particle, container::NBodyContainer)
    return dv
end

function TrixiParticles.write_u0!(u0, container::NBodyContainer)
    u0 .= container.initial_coordinates

    return u0
end

function TrixiParticles.write_v0!(v0, container::NBodyContainer)
    v0 .= container.initial_velocity

    return v0
end

# NHS update
function TrixiParticles.nhs_coords(container::NBodyContainer,
                                   neighbor::NBodyContainer, u)
    return u
end

function TrixiParticles.compact_support(container::NBodyContainer,
                                        neighbor::NBodyContainer)
    # There is no cutoff. All particles interact with each other.
    return Inf
end

function TrixiParticles.interact!(dv, v_particle_container, u_particle_container,
                                  v_neighbor_container, u_neighbor_container,
                                  neighborhood_search,
                                  particle_container::NBodyContainer,
                                  neighbor_container::NBodyContainer)
    @unpack mass, G = neighbor_container

    container_coords = TrixiParticles.current_coordinates(u_particle_container,
                                                          particle_container)

    neighbor_coords = TrixiParticles.current_coordinates(u_neighbor_container,
                                                         neighbor_container)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    TrixiParticles.for_particle_neighbor(particle_container, neighbor_container,
                                         container_coords, neighbor_coords,
                                         neighborhood_search) do particle, neighbor,
                                                                 pos_diff, distance
        # Only consider particles with a distance > 0.
        distance < sqrt(eps()) && return

        # Original version
        # dv = -G * mass[neighbor] * pos_diff / norm(pos_diff)^3

        # Multiplying by pos_diff later makes this slightly faster
        # Multiplying by (1 / norm) is also faster than dividing by norm
        tmp = -G * mass[neighbor] * (1 / distance^3)

        @inbounds for i in 1:ndims(particle_container)
            dv[i, particle] += tmp * pos_diff[i]
        end
    end

    return dv
end

function energy(v_ode, u_ode, container, semi)
    @unpack mass = container

    e = zero(eltype(container))

    v = TrixiParticles.wrap_v(v_ode, 1, container, semi)
    u = TrixiParticles.wrap_u(u_ode, 1, container, semi)

    for particle in TrixiParticles.eachparticle(container)
        e += 0.5 * mass[particle] *
             sum(TrixiParticles.current_velocity(v, container, particle) .^ 2)

        particle_coords = TrixiParticles.current_coords(u, container, particle)
        for neighbor in (particle + 1):TrixiParticles.nparticles(container)
            neighbor_coords = TrixiParticles.current_coords(u, container, neighbor)

            pos_diff = particle_coords - neighbor_coords
            distance = norm(pos_diff)

            e -= mass[particle] * mass[neighbor] / distance
        end
    end

    return e
end

TrixiParticles.vtkname(container::NBodyContainer) = "n-body"

function TrixiParticles.write2vtk!(vtk, v, u, t, container::NBodyContainer)
    @unpack mass = container

    vtk["velocity"] = v
    vtk["mass"] = mass

    return vtk
end

function Base.show(io::IO, container::NBodyContainer)
    print(io, "NBodyContainer{", ndims(container), "}() with ")
    print(io, TrixiParticles.nparticles(container), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", container::NBodyContainer)
    if get(io, :compact, false)
        show(io, container)
    else
        TrixiParticles.summary_header(io, "NBodyContainer{$(ndims(container))}")
        TrixiParticles.summary_line(io, "#particles", TrixiParticles.nparticles(container))
        TrixiParticles.summary_footer(io)
    end
end

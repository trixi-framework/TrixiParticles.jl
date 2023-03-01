using Pixie
using LinearAlgebra

struct NBodyContainer{NDIMS, ELTYPE <: Real} <: Pixie.ParticleContainer{NDIMS}
    initial_coordinates :: Array{ELTYPE, 2} # [dimension, particle]
    initial_velocity    :: Array{ELTYPE, 2} # [dimension, particle]
    mass                :: Array{ELTYPE, 1} # [particle]
    G                   :: ELTYPE

    function NBodyContainer(coordinates, velocities, masses, G)
        new{size(coordinates, 1), eltype(coordinates)}(coordinates, velocities, masses, G)
    end
end

@inline function Pixie.add_acceleration!(dv, particle, container::NBodyContainer)
    return dv
end

function Pixie.write_u0!(u0, container::NBodyContainer)
    u0 .= container.initial_coordinates

    return u0
end

function Pixie.write_v0!(v0, container::NBodyContainer)
    v0 .= container.initial_velocity

    return v0
end

# NHS update
function Pixie.update!(neighborhood_search, u,
                       container::NBodyContainer,
                       neighbor::NBodyContainer)
    Pixie.update!(neighborhood_search, u, neighbor)
end

function Pixie.interact!(dv, v_particle_container, u_particle_container,
                         v_neighbor_container, u_neighbor_container,
                         neighborhood_search,
                         particle_container::NBodyContainer,
                         neighbor_container::NBodyContainer)
    @unpack mass, G = neighbor_container

    for particle in Pixie.each_moving_particle(particle_container)
        particle_coords = Pixie.current_coords(u_particle_container,
                                               particle_container, particle)

        for neighbor in Pixie.eachneighbor(particle_coords, neighborhood_search)
            neighbor_coords = Pixie.current_coords(u_neighbor_container,
                                                   neighbor_container, neighbor)

            pos_diff = particle_coords - neighbor_coords
            distance = norm(pos_diff)

            if sqrt(eps()) < distance
                # Original version
                # dv = -G * mass[neighbor] * pos_diff / norm(pos_diff)^3

                # Multiplying by pos_diff later makes this slightly faster
                # Multiplying by (1 / norm) is also faster than dividing by norm
                tmp = -G * mass[neighbor] * (1 / distance^3)

                for i in 1:ndims(particle_container)
                    dv[i, particle] += tmp * pos_diff[i]
                end
            end
        end
    end

    return dv
end

function energy(v_ode, u_ode, container, semi)
    @unpack mass = container

    e = zero(eltype(container))

    v = Pixie.wrap_v(v_ode, 1, container, semi)
    u = Pixie.wrap_u(u_ode, 1, container, semi)

    for particle in Pixie.eachparticle(container)
        e += 0.5 * mass[particle] * sum(Pixie.current_velocity(v, container, particle) .^ 2)

        particle_coords = Pixie.current_coords(u, container, particle)
        for neighbor in (particle + 1):Pixie.nparticles(container)
            neighbor_coords = Pixie.current_coords(u, container, neighbor)

            pos_diff = particle_coords - neighbor_coords
            distance = norm(pos_diff)

            e -= mass[particle] * mass[neighbor] / distance
        end
    end

    return e
end

function (extract_quantities::Pixie.ExtractQuantities)(v, u, container::NBodyContainer)
    @unpack mass = container

    result = Dict{Symbol, Array{Float64}}(
                                          # Note that we have to allocate here and can't use views.
                                          # See https://diffeq.sciml.ai/stable/features/callback_library/#saving_callback.
                                          :coordinates => copy(u),
                                          :velocity => copy(v),
                                          :mass => copy(mass))

    return "n-body", result
end

function Base.show(io::IO, container::NBodyContainer)
    print(io, "NBodyContainer{", ndims(container), "}() with ")
    print(io, Pixie.nparticles(container), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", container::NBodyContainer)
    if get(io, :compact, false)
        show(io, container)
    else
        Pixie.summary_header(io, "NBodyContainer{$(ndims(container))}")
        Pixie.summary_line(io, "#particles", Pixie.nparticles(container))
        Pixie.summary_footer(io)
    end
end

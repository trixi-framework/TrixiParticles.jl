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

@inline function Pixie.add_acceleration!(du, particle, container::NBodyContainer)
    return du
end

function Pixie.write_variables!(u0, container::NBodyContainer)
    @unpack initial_coordinates, initial_velocity = container

    for particle in Pixie.eachparticle(container)
        # Write particle coordinates
        for dim in 1:ndims(container)
            u0[dim, particle] = initial_coordinates[dim, particle]
        end

        # Write particle velocities
        for dim in 1:ndims(container)
            u0[dim + ndims(container), particle] = initial_velocity[dim, particle]
        end
    end

    return u0
end

function Pixie.interact!(du, u_particle_container, u_neighbor_container,
                         neighborhood_search,
                         particle_container::NBodyContainer,
                         neighbor_container::NBodyContainer)
    @unpack mass, G = neighbor_container

    for particle in Pixie.each_moving_particle(particle_container)
        particle_coords = Pixie.get_current_coords(particle, u_particle_container,
                                                   particle_container)

        for neighbor in Pixie.eachneighbor(particle_coords, neighborhood_search)
            neighbor_coords = Pixie.get_current_coords(neighbor, u_neighbor_container,
                                                       neighbor_container)

            pos_diff = particle_coords - neighbor_coords
            distance = norm(pos_diff)

            if sqrt(eps()) < distance
                # Original version
                # dv = -G * mass[neighbor] * pos_diff / norm(pos_diff)^3

                # Multiplying by pos_diff later makes this slightly faster
                # Multiplying by (1 / norm) is also faster than dividing by norm
                tmp = -G * mass[neighbor] * (1 / distance^3)

                for i in 1:ndims(particle_container)
                    du[ndims(particle_container) + i, particle] += tmp * pos_diff[i]
                end
            end
        end
    end

    return du
end

function energy(u_ode, container, semi)
    @unpack mass = container

    e = zero(eltype(container))

    u = Pixie.wrap_array(u_ode, 1, container, semi)

    for particle in Pixie.eachparticle(container)
        e += 0.5 * mass[particle] * sum(Pixie.get_particle_vel(particle, u, container) .^ 2)

        particle_coords = Pixie.get_current_coords(particle, u, container)
        for neighbor in (particle + 1):Pixie.nparticles(container)
            neighbor_coords = Pixie.get_current_coords(neighbor, u, container)

            pos_diff = particle_coords - neighbor_coords
            distance = norm(pos_diff)

            e -= mass[particle] * mass[neighbor] / distance
        end
    end

    return e
end

function (extract_quantities::Pixie.ExtractQuantities)(u, container::NBodyContainer)
    @unpack mass = container

    result = Dict{Symbol, Array{Float64}}(
                                          # Note that we have to allocate here and can't use views.
                                          # See https://diffeq.sciml.ai/stable/features/callback_library/#saving_callback.
                                          :coordinates => u[1:ndims(container), :],
                                          :velocity => u[(ndims(container) + 1):(2 * ndims(container)),
                                                         :],
                                          :mass => copy(mass))

    return "n-body", result
end

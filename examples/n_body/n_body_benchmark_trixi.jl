#
# An n-body benchmark from "The Computer Language Benchmarks Game" translated to TrixiParticles.
# This example does not benefit from using multiple threads.
# Multithreading is disabled below.
#

using TrixiParticles
using Printf
using Polyester

include("n_body_container.jl")

# Redefine interact in a more optimized way.
function TrixiParticles.interact!(du, u_particle_container, u_neighbor_container,
                                  neighborhood_search,
                                  particle_container::NBodyContainer,
                                  neighbor_container::NBodyContainer)
    @unpack mass, G = neighbor_container

    for particle in TrixiParticles.each_moving_particle(particle_container)
        particle_coords = TrixiParticles.current_coords(u_particle_container,
                                                        particle_container, particle)

        # This makes `interact!` about 20% faster than `eachneighbor` with `particle < neighbor`.
        # Note that this doesn't work if we have multiple containers.
        for neighbor in (particle + 1):TrixiParticles.nparticles(neighbor_container)
            neighbor_coords = TrixiParticles.current_coords(u_neighbor_container,
                                                            neighbor_container, neighbor)
            pos_diff = particle_coords - neighbor_coords

            # Multiplying by pos_diff later makes this slightly faster.
            # Multiplying by (1 / norm) is also faster than dividing by norm.
            tmp = -G * (1 / norm(pos_diff)^3)
            tmp1 = mass[neighbor] * tmp
            tmp2 = mass[particle] * tmp

            for i in 1:ndims(particle_container)
                j = i + ndims(particle_container)
                # This is slightly faster than du[...] += tmp1 * pos_diff[i]
                du[j, particle] = muladd(tmp1, pos_diff[i], du[j, particle])
                du[j, neighbor] = muladd(tmp2, -pos_diff[i], du[j, neighbor])
            end
        end
    end

    return du
end

# ==========================================================================================
# ==== Container

const SOLAR_MASS = 4 * pi * pi
const DAYS_PER_YEAR = 365.24
coordinates = [0.0 4.84143144246472090e+0 8.34336671824457987e+0 1.28943695621391310e+1 1.53796971148509165e+1;
               0.0 -1.16032004402742839e+0 4.12479856412430479e+0 -1.51111514016986312e+1 -2.59193146099879641e+1;
               0.0 -1.03622044471123109e-1 -4.03523417114321381e-1 -2.23307578892655734e-1 1.79258772950371181e-1]

velocities = [0.0 1.66007664274403694e-3 -2.76742510726862411e-3 2.96460137564761618e-3 2.68067772490389322e-3;
              0.0 7.69901118419740425e-3 4.99852801234917238e-3 2.37847173959480950e-3 1.62824170038242295e-3;
              0.0 -6.90460016972063023e-5 2.30417297573763929e-5 -2.96589568540237556e-5 -9.51592254519715870e-5] *
             DAYS_PER_YEAR

masses = [
    1.0, 9.54791938424326609e-4, 2.85885980666130812e-4, 4.36624404335156298e-5,
    5.15138902046611451e-5,
] * SOLAR_MASS

# Offset sun momentum
velocities[:, 1] = -velocities[:, 2:end] * masses[2:end] / SOLAR_MASS

G = 1.0
particle_container = NBodyContainer(coordinates, velocities, masses, G)

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(particle_container)

# This is significantly faster than using OrdinaryDiffEq.
function symplectic_euler!(velocities, coordinates, semi)
    v = vec(velocities)
    u = vec(coordinates)
    dv = copy(v)
    du = copy(u)

    @time for _ in 1:50_000_000
        TrixiParticles.kick!(dv, v, u, semi, 0.0)

        @inbounds for i in eachindex(v)
            v[i] += 0.01 * dv[i]
        end

        TrixiParticles.drift!(du, v, u, semi, 0.0)

        @inbounds for i in eachindex(u)
            u[i] += 0.01 * du[i]
        end
    end
end

# One RHS evaluation is so fast that timers make it multiple times slower.
TrixiParticles.TimerOutputs.disable_debug_timings(TrixiParticles)

@printf("%.9f\n", energy(velocities, coordinates, particle_container, semi))

# Disable multithreading, since it adds a significant overhead for this small problem.
disable_polyester_threads() do
    symplectic_euler!(velocities, coordinates, semi)
end

@printf("%.9f\n", energy(velocities, coordinates, particle_container, semi))

# Enable timers again.
TrixiParticles.TimerOutputs.enable_debug_timings(TrixiParticles)

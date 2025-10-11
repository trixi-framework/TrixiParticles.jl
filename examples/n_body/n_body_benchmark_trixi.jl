# ==========================================================================================
# An n-body benchmark from "The Computer Language Benchmarks Game" translated to TrixiParticles.
# This example does not benefit from using multiple threads.
# Multithreading is disabled below.
# ==========================================================================================

using TrixiParticles
using Printf
using Polyester

include("n_body_system.jl")

# Redefine interact in a more optimized way
function TrixiParticles.interact!(dv, v_particle_system, u_particle_system,
                                  v_neighbor_system, u_neighbor_system,
                                  neighborhood_search,
                                  particle_system::NBodySystem,
                                  neighbor_system::NBodySystem)
    (; mass, G) = neighbor_system

    for particle in TrixiParticles.each_integrated_particle(particle_system)
        particle_coords = TrixiParticles.current_coords(u_particle_system,
                                                        particle_system, particle)

        # This makes `interact!` about 20% faster than looping over all particles
        # and checking for `particle < neighbor`.
        # Note that this doesn't work if we have multiple systems.
        for neighbor in (particle + 1):TrixiParticles.nparticles(neighbor_system)
            neighbor_coords = TrixiParticles.current_coords(u_neighbor_system,
                                                            neighbor_system, neighbor)
            pos_diff = particle_coords - neighbor_coords

            # Multiplying by pos_diff later makes this slightly faster.
            # Multiplying by (1 / norm) is also faster than dividing by norm.
            tmp = -G * (1 / norm(pos_diff)^3)
            tmp1 = mass[neighbor] * tmp
            tmp2 = mass[particle] * tmp

            @inbounds for i in 1:ndims(particle_system)
                # This is slightly faster than dv[...] += tmp1 * pos_diff[i]
                dv[i, particle] = muladd(tmp1, pos_diff[i], dv[i, particle])
                dv[i, neighbor] = muladd(tmp2, -pos_diff[i], dv[i, neighbor])
            end
        end
    end

    return dv
end

# ==========================================================================================
# ==== Systems

const SOLAR_MASS = 4 * pi * pi
const DAYS_PER_YEAR = 365.24
coordinates = [0.0 4.84143144246472090e+0 8.34336671824457987e+0 1.28943695621391310e+1 1.53796971148509165e+1;
               0.0 -1.16032004402742839e+0 4.12479856412430479e+0 -1.51111514016986312e+1 -2.59193146099879641e+1;
               0.0 -1.03622044471123109e-1 -4.03523417114321381e-1 -2.23307578892655734e-1 1.79258772950371181e-1]

velocity = [0.0 1.66007664274403694e-3 -2.76742510726862411e-3 2.96460137564761618e-3 2.68067772490389322e-3;
            0.0 7.69901118419740425e-3 4.99852801234917238e-3 2.37847173959480950e-3 1.62824170038242295e-3;
            0.0 -6.90460016972063023e-5 2.30417297573763929e-5 -2.96589568540237556e-5 -9.51592254519715870e-5] *
           DAYS_PER_YEAR

masses = [
    1.0, 9.54791938424326609e-4, 2.85885980666130812e-4, 4.36624404335156298e-5,
    5.15138902046611451e-5
] * SOLAR_MASS

# Offset sun momentum
velocity[:, 1] = -velocity[:, 2:end] * masses[2:end] / SOLAR_MASS

initial_condition = InitialCondition(; coordinates, velocity, density=1.0, mass=masses)

G = 1.0
particle_system = NBodySystem(initial_condition, G)

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(particle_system, neighborhood_search=nothing)

# This is significantly faster than using OrdinaryDiffEq.
function symplectic_euler!(velocity, coordinates, semi)
    v = vec(velocity)
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

    return v, u
end

# One RHS evaluation is so fast that timers make it multiple times slower.
# Disabling timers throws a warning, which we suppress here in order to make the tests pass.
# For some reason, this only works with a file and not with devnull. See issue #332.
filename = tempname()
open(filename, "w") do f
    redirect_stderr(f) do
        TrixiParticles.disable_debug_timings()
    end
end

@printf("%.9f\n", energy(velocity, coordinates, particle_system, semi))

# Disable multithreading, since it adds a significant overhead for this small problem.
disable_polyester_threads() do
    symplectic_euler!(velocity, coordinates, semi)
end

@printf("%.9f\n", energy(velocity, coordinates, particle_system, semi))

# Enable timers again
open(filename, "w") do f
    redirect_stderr(f) do
        TrixiParticles.enable_debug_timings()
    end
end

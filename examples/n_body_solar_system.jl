using Pixie
using OrdinaryDiffEq
using UnPack: @unpack
using StaticArrays: SVector
using LinearAlgebra

struct NBodyContainer{NDIMS, ELTYPE<:Real} <: Pixie.ParticleContainer{NDIMS}
    initial_coordinates ::Array{ELTYPE, 2} # [dimension, particle]
    initial_velocity    ::Array{ELTYPE, 2} # [dimension, particle]
    mass                ::Array{ELTYPE, 1} # [particle]
    G                   ::ELTYPE
    softening           ::ELTYPE
    acceleration        ::SVector{NDIMS, ELTYPE} # Background acceleration (always zero)

    function NBodyContainer(coordinates, velocities, masses, G, softening)
        new{size(coordinates, 1), eltype(coordinates)}(coordinates, velocities, masses, G, softening,
                                                       ntuple(_ -> 0.0, size(coordinates, 1)))
    end
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

# mass-particle to mass-particle interaction
function Pixie.interact!(du, u_particle_container, u_neighbor_container, neighborhood_search,
                   particle_container::NBodyContainer,
                   neighbor_container::NBodyContainer)

    @unpack mass,G,softening = neighbor_container

    # Pixie.@threaded for particle in Pixie.each_moving_particle(particle_container) # particle i
    for particle in Pixie.each_moving_particle(particle_container) # particle i

        particle_coords = Pixie.get_current_coords(particle, u_particle_container, particle_container)
        for neighbor in Pixie.eachneighbor(particle_coords, neighborhood_search) # particle j
            neighbor_coords = Pixie.get_current_coords(neighbor, u_neighbor_container, neighbor_container)

            pos_diff = particle_coords - neighbor_coords
            distance = norm(pos_diff)
            distance = sqrt(distance^2 + softening^2)

            neighbor_mass = mass[neighbor]
            if sqrt(eps()) < distance
                dv = -G * neighbor_mass * pos_diff / distance^3

                for i in 1:ndims(particle_container)
                    du[ndims(particle_container) + i, particle] += dv[i]
                end
            end
        end
    end

    return du
end


function (extract_quantities::Pixie.ExtractQuantities)(u, container::NBodyContainer)
    @unpack mass = container

    result = Dict{Symbol, Array{Float64}}(
        # Note that we have to allocate here and can't use views.
        # See https://diffeq.sciml.ai/stable/features/callback_library/#saving_callback.
        :coordinates    => u[1:ndims(container), :],
        :velocity       => u[(ndims(container)+1):(2*ndims(container)), :],
        :mass           => copy(mass)
    )

    return "n-body", result
end

#coordinates = [
#    0.0 1.0 -1;
#    0.0 0.0 0
#]
#velocities = [
#    0 0 0;
#    0 1 -1
#]
#
#masses = [20.0, 1.0, 1.0]

#const SOLAR_MASS = 4 * pi * pi
#const DAYS_PER_YEAR = 365.24
#coordinates = [
#    0.0 4.84143144246472090e+0  8.34336671824457987e+0  1.28943695621391310e+1  1.53796971148509165e+1;
#    0.0 -1.16032004402742839e+0  4.12479856412430479e+0 -1.51111514016986312e+1 -2.59193146099879641e+1;
#    0.0 -1.03622044471123109e-1 -4.03523417114321381e-1 -2.23307578892655734e-1 1.79258772950371181e-1
#]
#
#velocities = [
#    0.0 1.66007664274403694e-3 -2.76742510726862411e-3 2.96460137564761618e-3 2.68067772490389322e-3;
#    0.0 7.69901118419740425e-3 4.99852801234917238e-3 2.37847173959480950e-3 1.62824170038242295e-3;
#    0.0 -6.90460016972063023e-5 2.30417297573763929e-5 -2.96589568540237556e-5 -9.51592254519715870e-5;
#] * DAYS_PER_YEAR
#
#masses = [
#    1.0, 9.54791938424326609e-4, 2.85885980666130812e-4, 4.36624404335156298e-5, 5.15138902046611451e-5
#] * SOLAR_MASS
#
## Offset sun momentum
#velocities[:, 1] = -velocities[:, 2:end] * masses[2:end] / SOLAR_MASS
#
# infos from: https://de.mathworks.com/help/sm/ug/model-planet-orbit-due-to-gravity.html;jsessionid=1c983e477bd0e1231ad25ef4c5d1
coordinates = [
              5.5850e+08 5.1979e+10 -1.5041e+10 -1.1506e+09 -4.8883e+10 -8.1142e+11 -4.2780e+11 2.7878e+12 4.2097e+12;
              5.5850e+08 7.6928e+09 9.7080e+10 -1.3910e+11 -1.9686e+11 4.5462e+10 -1.3353e+12 9.9509e+11 -1.3834e+12;
              5.5850e+08 -1.2845e+09 4.4635e+10 -6.0330e+10 -8.8994e+10 3.9229e+10 -5.3311e+11 3.9639e+08 -6.7105e+11
              ]
velocities = [
             -1.4663 -1.5205e+04 -3.4770e+04 2.9288e+04 2.4533e+04 -1.0724e+03 8.7288e+03 -2.4913e+03 1.8271e+03;
             11.1238 4.4189e+04 -5.5933e+03 -398.5759 -2.7622e+03 -1.1422e+04 -2.4369e+03 5.5197e+03 4.7731e+03;
             4.8370 2.5180e+04 -316.8994 -172.5873 -1.9295e+03 -4.8696e+03 -1.3824e+03 2.4527e+03 1.9082e+03
             ]

masses = [
          1.99e30, 3.30e23, 4.87e24, 5.97e24, 6.42e23, 1.90e27, 5.68e26, 8.68e25, 1.02e26
         ]

G = 6.6743e-11
softening = 0.0
particle_container = NBodyContainer(coordinates, velocities, masses,G,softening)

semi = Semidiscretization(particle_container)

years = 100
tend = years * 365 * 24 * 3600
day = 24 * 3600
tspan = (0.0, tend)
ode = semidiscretize(semi, tspan)

alive_callback = AliveCallback(alive_interval=1000)

saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:10*day:tend)

callbacks = CallbackSet(alive_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
#sol = solve(ode, RDPK3SpFSAL49(thread=OrdinaryDiffEq.True()),
sol = solve(ode, Trapezoid(nlsolve = NLFunctional()),
            dt=1e-2, # Initial guess of the time step to prevent too large guesses
            abstol=1.0e-6, reltol=1.0e-6, # Tighter tolerance to prevent instabilities
            save_everystep=false, callback=callbacks,
            maxiters=10_000_000);


# sol = solve(ode, Euler(),
#             dt=0.01,
#             save_everystep=false, callback=callbacks,
#             maxiters=60_000_000);

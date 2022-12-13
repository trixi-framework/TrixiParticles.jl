using Pixie
using OrdinaryDiffEq
using UnPack: @unpack
using StaticArrays: SVector

struct NBodyContainer{NDIMS, ELTYPE<:Real} <: Pixie.ParticleContainer{NDIMS}
    initial_coordinates ::Array{ELTYPE, 2} # [dimension, particle]
    initial_velocity    ::Array{ELTYPE, 2} # [dimension, particle]
    mass                ::Array{ELTYPE, 1} # [particle]
    G                   ::ELTYPE
    acceleration        ::SVector{NDIMS, ELTYPE} # Background acceleration (always zero)

    function NBodyContainer(coordinates, velocities, masses, G)
        new{size(coordinates, 1), eltype(coordinates)}(coordinates, velocities, masses, G,
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

coordinates = [
    0 1 -1;
    0 0 0
]
velocities = [
    0 0 0;
    0 1 -1
]
masses = [1000.0, 1.0, 1.0]
G = 1 # 6.67e-11
particle_container = NBodyContainer(coordinates, velocities, masses, G)

semi = Semidiscretization(particle_container)

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

alive_callback = AliveCallback(alive_interval=1)

# saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:0.02:20.0)

callbacks = CallbackSet(alive_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
sol = solve(ode, RDPK3SpFSAL49(thread=OrdinaryDiffEq.True()),
            dt=1e-4, # Initial guess of the time step to prevent too large guesses
            abstol=1.0e-4, reltol=1.0e-4, # Tighter tolerance to prevent instabilities
            save_everystep=false, callback=callbacks);

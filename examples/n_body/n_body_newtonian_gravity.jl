# ==========================================================================================
# A minimal two-body simulation using the `NewtonianGravity` force model.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEqSymplecticRK

include("n_body_system.jl")

# ==========================================================================================
# ==== Systems

coordinates = [-0.5 0.5;
               0.0 0.0]

velocity = [0.0 0.0;
            -sqrt(0.5) sqrt(0.5)]

masses = [1.0, 1.0]

initial_condition = InitialCondition(; coordinates, velocity, density=1.0, mass=masses)

gravity = NewtonianGravity(; gravitational_constant=1.0)
particle_system = NBodySystem(initial_condition, gravity)

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(particle_system, neighborhood_search=nothing)

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

sol = solve(ode, SymplecticEuler(), dt=0.001, save_everystep=false);

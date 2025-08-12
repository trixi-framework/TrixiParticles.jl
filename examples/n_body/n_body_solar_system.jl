# ==========================================================================================
# An n-body simulation of the solar system.
# This example does not benefit from using multiple threads due to the small problem size.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq
using Printf

include("n_body_system.jl")

# ==========================================================================================
# ==== Systems

# Data from https://de.mathworks.com/help/sm/ug/model-planet-orbit-due-to-gravity.html;jsessionid=1c983e477bd0e1231ad25ef4c5d1
coordinates = [
    5.585e+8 5.1979e+10 -1.5041e+10 -1.1506e+9 -4.8883e+10 -8.1142e+11 -4.278e+11 2.7878e+12 4.2097e+12;
    5.585e+8 7.6928e+9 9.708e+10 -1.391e+11 -1.9686e+11 4.5462e+10 -1.3353e+12 9.9509e+11 -1.3834e+12;
    5.585e+8 -1.2845e+9 4.4635e+10 -6.033e+10 -8.8994e+10 3.9229e+10 -5.3311e+11 3.9639e+8 -6.7105e+11
]
velocity = [
    -1.4663 -1.5205e+4 -3.477e+4 2.9288e+4 2.4533e+4 -1.0724e+3 8.7288e+3 -2.4913e+3 1.8271e+3;
    11.1238 4.4189e+4 -5.5933e+3 -398.5759 -2.7622e+3 -1.1422e+4 -2.4369e+3 5.5197e+3 4.7731e+3;
    4.837 2.518e+4 -316.8994 -172.5873 -1.9295e+3 -4.8696e+3 -1.3824e+3 2.4527e+3 1.9082e+3
]

masses = [
    1.99e30, 3.3e23, 4.87e24, 5.97e24, 6.42e23, 1.9e27, 5.68e26, 8.68e25, 1.02e26,
]

initial_condition = InitialCondition(; coordinates, velocity, density = 1.0, mass = masses)

G = 6.6743e-11
particle_system = NBodySystem(initial_condition, G)

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(particle_system, neighborhood_search = nothing)

day = 24 * 3600.0
year = 365day
tspan = (0.0, 10year)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval = 100000)
saving_callback = SolutionSavingCallback(dt = 10day, max_coordinates = Inf)

callbacks = CallbackSet(info_callback, saving_callback)

# One RHS evaluation is so fast that timers make it multiple times slower.
# Disabling timers throws a warning, which we suppress here in order to make the tests pass.
# For some reason, this only works with a file and not with devnull. See issue #332.
filename = tempname()
open(filename, "w") do f
    redirect_stderr(f) do
        TrixiParticles.disable_debug_timings()
    end
end

sol = solve(
    ode, SymplecticEuler(),
    dt = 1.0e5,
    save_everystep = false, callback = callbacks
);

@printf("%.9e\n", energy(ode.u0.x..., particle_system, semi))
@printf("%.9e\n", energy(sol.u[end].x..., particle_system, semi))

# Enable timers again
open(filename, "w") do f
    redirect_stderr(f) do
        TrixiParticles.enable_debug_timings()
    end
end

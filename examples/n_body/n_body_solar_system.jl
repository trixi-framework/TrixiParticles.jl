#
# An n-body simulation of the solar system
#
using Pixie
using OrdinaryDiffEq
using Printf

include("n_body_container.jl")

# Data from https://de.mathworks.com/help/sm/ug/model-planet-orbit-due-to-gravity.html;jsessionid=1c983e477bd0e1231ad25ef4c5d1
coordinates = [5.5850e+08 5.1979e+10 -1.5041e+10 -1.1506e+09 -4.8883e+10 -8.1142e+11 -4.2780e+11 2.7878e+12 4.2097e+12;
               5.5850e+08 7.6928e+09 9.7080e+10 -1.3910e+11 -1.9686e+11 4.5462e+10 -1.3353e+12 9.9509e+11 -1.3834e+12;
               5.5850e+08 -1.2845e+09 4.4635e+10 -6.0330e+10 -8.8994e+10 3.9229e+10 -5.3311e+11 3.9639e+08 -6.7105e+11]
velocities = [-1.4663 -1.5205e+04 -3.4770e+04 2.9288e+04 2.4533e+04 -1.0724e+03 8.7288e+03 -2.4913e+03 1.8271e+03;
              11.1238 4.4189e+04 -5.5933e+03 -398.5759 -2.7622e+03 -1.1422e+04 -2.4369e+03 5.5197e+03 4.7731e+03;
              4.8370 2.5180e+04 -316.8994 -172.5873 -1.9295e+03 -4.8696e+03 -1.3824e+03 2.4527e+03 1.9082e+03]

masses = [
    1.99e30, 3.30e23, 4.87e24, 5.97e24, 6.42e23, 1.90e27, 5.68e26, 8.68e25, 1.02e26,
]

G = 6.6743e-11
particle_container = NBodyContainer(coordinates, velocities, masses, G)

semi = Semidiscretization(particle_container)

day = 24 * 3600.0
year = 365day
tspan = (0.0, 10year)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=100000)

saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:(10day):tspan[end])

callbacks = CallbackSet(summary_callback, alive_callback, saving_callback)

# One RHS evaluation is so fast that timers make it multiple times slower
TimerOutputs.disable_debug_timings(Pixie)

sol = solve(ode, SymplecticEuler(),
            dt=1.0e5,
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()

@printf("%.9e\n", energy(ode.u0, particle_container, semi))
@printf("%.9e\n", energy(sol[end], particle_container, semi))

# Enable timers again
TimerOutputs.enable_debug_timings(Pixie)

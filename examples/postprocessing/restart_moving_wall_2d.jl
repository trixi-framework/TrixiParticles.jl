using TrixiParticles
using OrdinaryDiffEq

# Start Simulation
trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "moving_wall_2d.jl"),
              tspan=(0.0, 1.0))

save_jld2(semi, sol)

# Restart Simulation at `tspan[2]`
t_end = 2.0
ode_restart = load_jld2(t_end)

saving_callback = SolutionSavingCallback(dt=0.02, prefix="restart")
callbacks = CallbackSet(info_callback, saving_callback)

sol_restart = solve(ode_restart, RDPK3SpFSAL35(),
                    abstol=1e-6,
                    reltol=1e-4,
                    dtmax=1e-2,
                    save_everystep=false, callback=callbacks);

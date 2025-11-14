using TrixiParticles

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "moving_wall_2d.jl"), sol=nothing)

#save_config(semi, sol; output_directory="out", filename="config")


# movement_function(x, t) = x + SVector(0.5 * t^2, 0.0)

# is_moving(t) = t < 1.5

tspan = [1.0, 2.5]

ode_restart = load_ode(tspan; input_directory="out", filename="config")

# info_callback = InfoCallback(interval=100)
# saving_callback = SolutionSavingCallback(dt=0.02, prefix="2")
# callbacks = CallbackSet(info_callback, saving_callback)

sol_restart = solve(ode_restart, RDPK3SpFSAL35(),
                    abstol=1e-6,
                    reltol=1e-4,
                    dtmax=1e-2,
                    save_everystep=false, callback=callbacks);

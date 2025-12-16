# ==========================================================================================
# Restart Example
#
# This example demonstrates how to use the `save_checkpoint` and `load_checkpoint` functions
# in TrixiParticles.jl to restart a simulation at a specific timepoint.
# A moving wall simulation is used as the base.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# Start Simulation
trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "moving_wall_2d.jl"),
              tspan=(0.0, 1.0))

file_checkpoint = save_checkpoint(sol)

sol_checkpoint = load_checkpoint(file_checkpoint)

tspan = (1.0, 2.0)
ode_checkpoint = semidiscretize_from_checkpoint(sol_checkpoint, tspan)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="restart")
callbacks = CallbackSet(info_callback, saving_callback)

sol_restart = solve(ode_checkpoint, RDPK3SpFSAL35(),
                    abstol=1e-6,
                    reltol=1e-4,
                    dtmax=1e-2,
                    save_everystep=false, callback=callbacks);

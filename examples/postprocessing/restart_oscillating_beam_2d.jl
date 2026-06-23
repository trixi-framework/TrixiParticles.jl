# ==========================================================================================
# Restart Example: Oscillating Beam 2D
#
# This example demonstrates how to restart a simulation.
# We first run a simulation of oscillating beam up to t=1.0s, then restart from the
# saved state and continue the simulation until t=2.0s.
# ==========================================================================================
using TrixiParticles

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "structure", "oscillating_beam_2d.jl"),
              tspan=(0.0, 1.0))

# Get latest iteration
iter = saving_callback.affect!.affect!.latest_saved_iter

restart_file = joinpath("out", "structure_1_$iter.vtu")

ode_restart = semidiscretize(semi, (1.0, 2.0);
                             restart_with=restart_file)

saving_callback = SolutionSavingCallback(dt=0.02, prefix="restart")

callbacks = CallbackSet(info_callback, saving_callback)

sol_restart = solve(ode_restart, RDPK3SpFSAL35(), abstol=1e-5, reltol=1e-3, dtmax=1e-2,
                    save_everystep=false, callback=callbacks)

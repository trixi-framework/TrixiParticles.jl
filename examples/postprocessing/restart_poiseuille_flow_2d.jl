# ==========================================================================================
# Restart Example: Poiseuille Flow 2D
#
# This example demonstrates how to restart a simulation.
# We first run a simulation of 2D Poiseuille flow up to t=0.3s, then restart from the
# saved state and continue the simulation until t=0.6s.
# ==========================================================================================
using TrixiParticles

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "poiseuille_flow_2d.jl"),
              tspan=(0.0, 0.3), sound_speed_factor=10, particle_spacing=4e-5)

# Get latest iteration
iter = saving_callback.condition.index[] - 1

restart_file_fluid = joinpath("out", "fluid_1_$iter.vtu")
restart_file_open_boundary = joinpath("out", "open_boundary_1_$iter.vtu")
restart_file_boundary = joinpath("out", "boundary_1_$iter.vtu")

ode_restart = semidiscretize(semi, (0.3, 0.6);
                             restart_with=(restart_file_fluid,
                                           restart_file_open_boundary,
                                           restart_file_boundary))

saving_callback = SolutionSavingCallback(dt=0.02, prefix="restart")

callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback())

sol_restart = solve(ode_restart, RDPK3SpFSAL35(), abstol=1e-5, reltol=1e-3, dtmax=1e-2,
                    save_everystep=false, callback=callbacks)

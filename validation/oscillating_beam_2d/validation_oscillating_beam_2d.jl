# Results are compared to the results in:
#
# P.N. Sun, D. Le Touzé, A.-M. Zhang.
# "Study of a complex fluid-structure dam-breaking benchmark problem using a multi-phase SPH method with APR".
# In: Engineering Analysis with Boundary Elements 104 (2019), pages 240-258.
# https://doi.org/10.1016/j.enganabound.2019.03.033
# and
# Turek S , Hron J.
# "Proposal for numerical benchmarking of fluid-structure interaction between an elastic object and laminar incompressible flow."
# In: Fluid-structure interaction. Springer; 2006. p. 371–85 .
# https://doi.org/10.1007/3-540-34596-5_15

include("../validation_util.jl")
using TrixiParticles
using OrdinaryDiffEq
using JSON

tspan = (0, 10)

# `n_particles_beam_y = 5` means that the beam is 5 particles thick.
# This number is used to set the resolution of the simulation.
# It has to be odd, so that a particle is exactly in the middle of the tip of the beam.
# Use 5, 9, 21, 35 for validation.
# Note: 35 takes a very long time!
n_particles_beam_y = 5

# Overwrite `sol` assignment to skip time integration
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "structure", "oscillating_beam_2d.jl"),
              n_particles_y=n_particles_beam_y, sol=nothing, tspan=tspan,
              penalty_force=PenaltyForceGanzenmueller(alpha=0.01))

pp_callback = PostprocessCallback(; deflection_x, deflection_y, dt=0.01,
                                  output_directory="out",
                                  filename="validation_run_oscillating_beam_2d_$n_particles_beam_y",
                                  write_csv=false, write_file_interval=0)
info_callback = InfoCallback(interval=2500)

callbacks = CallbackSet(info_callback, pp_callback)

sol = solve(ode, RDPK3SpFSAL49(), abstol=1e-8, reltol=1e-6, dt=1e-5,
            save_everystep=false, callback=callbacks)

reference_file_name = joinpath(validation_dir(), "oscillating_beam_2d",
                               "validation_reference_$n_particles_beam_y.json")
run_file_name = joinpath("out",
                         "validation_run_oscillating_beam_2d_$n_particles_beam_y.json")

reference_data = JSON.parsefile(reference_file_name)
run_data = JSON.parsefile(run_file_name)

error_deflection_x = interpolated_mse(reference_data["deflection_x_structure_1"]["time"],
                                      reference_data["deflection_x_structure_1"]["values"],
                                      run_data["deflection_x_structure_1"]["time"],
                                      run_data["deflection_x_structure_1"]["values"])

error_deflection_y = interpolated_mse(reference_data["deflection_y_structure_1"]["time"],
                                      reference_data["deflection_y_structure_1"]["values"],
                                      run_data["deflection_y_structure_1"]["time"],
                                      run_data["deflection_y_structure_1"]["values"])

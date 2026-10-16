# Oscillating beam validation with the setup of `examples/structure/oscillating_beam_2d.jl`,
# which converges to the reference solution of the CSM3 benchmark by Turek and Hron (2006):
# - Particles are placed cell-centered, so that the particle volumes fill exactly
#   the beam with the given length and thickness.
# - The elastic beam is glued to the rigid cylinder, i.e., the elastic beam starts at the
#   cylinder surface and all particles inside the cylinder are clamped.
# See `validation_oscillating_beam_2d_oconnor.jl` for the setup
# of O'Connor and Rogers (2021), which does not converge to this reference solution.
#
# Results are compared to the results in:
#
# S. Turek and J. Hron.
# "Proposal for numerical benchmarking of fluid-structure interaction between an elastic
# object and laminar incompressible flow".
# In: Fluid-Structure Interaction, Springer, 2006, pp. 371-385.
# https://doi.org/10.1007/3-540-34596-5_15

include("../validation_util.jl")
using TrixiParticles
using OrdinaryDiffEqSymplecticRK
using JSON

tspan = (0, 10)

# `n_particles_beam_y = 5` means that the beam is 5 particles thick,
# i.e., `thickness / particle_spacing = n_particles_beam_y`.
# This number is used to set the resolution of the simulation.
# The reference files are available for 5, 9, 17, 33, 65, 129, 257, 513.
# The reference files for 129, 257, 513 only cover the first swing (`tspan = (0, 0.6)`).
n_particles_beam_y = 5

parallelization_backend = PolyesterBackend()

# Overwrite `sol` assignment to skip time integration.
trixi_include(@__MODULE__, joinpath(examples_dir(), "structure", "oscillating_beam_2d.jl");
              n_particles_y=n_particles_beam_y, sol=nothing, tspan, parallelization_backend)

# Track the deflection of the center of the tip of the beam.
# Tracking individual particles is not the correct approach, because particles are
# interpolation points, so the motion is reconstructed by an SPH interpolation over
# the surrounding particles. Note that `Semidiscretization` creates a deep copy of the
# structure system, which means we have to extract the new system from `semi`.
tip_position = (tip_x, cylinder_center[2])
deflection_x = StructureMotionCalculator(semi.systems[1], semi, tip_position,
                                         quantity=motion -> motion.displacement[1])
deflection_y = StructureMotionCalculator(semi.systems[1], semi, tip_position,
                                         quantity=motion -> motion.displacement[2])

pp_callback = PostprocessCallback(; deflection_x, deflection_y, dt=0.01,
                                  output_directory="out",
                                  filename="validation_run_oscillating_beam_2d_turek_$n_particles_beam_y",
                                  write_csv=false, write_file_interval=10)
info_callback = InfoCallback(interval=10000)

callbacks = CallbackSet(info_callback, pp_callback, stepsize_callback)

# `dt` is overwritten by the `StepsizeCallback`.
sol = solve(ode, VelocityVerlet(), dt=1.0, maxiters=10^8,
            save_everystep=false, callback=callbacks)

# Compare against the reference results generated with TrixiParticles.jl,
# NOT the results from Turek & Hron (2006), which are used in the convergence plot.
reference_file_name = joinpath(validation_dir(), "oscillating_beam_2d",
                               "validation_reference_turek_$n_particles_beam_y.json")
run_file_name = joinpath("out",
                         "validation_run_oscillating_beam_2d_turek_$n_particles_beam_y.json")

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

println("Validation results for oscillating beam 2D (Turek setup) with $n_particles_beam_y particles in beam thickness:")
println("  MSE deflection x: $error_deflection_x")
println("  MSE deflection y: $error_deflection_y")

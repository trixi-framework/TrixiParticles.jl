# Oscillating beam validation with the setup of O'Connor and Rogers (2021):
# - Particles are placed on the shell of the beam (`place_on_shell=true`), so that the
#   discrete beam has a thickness of `thickness + particle_spacing`.
# - The beam continues inside the clamping cylinder, where it is only bonded to the clamped
#   cylinder particles. This makes the setup more compliant than the rigid cylinder in the
#   benchmark by Turek and Hron (2006).
# Therefore, this setup does not converge to the reference solution by Turek and Hron.
# See `validation_oscillating_beam_2d_turek.jl` and `examples/structure/oscillating_beam_2d.jl`
# for a setup that does.
#
# J. O'Connor and B.D. Rogers.
# "A fluid-structure interaction model for free-surface flows and flexible structures using
# smoothed particle hydrodynamics on a GPU".
# In: Journal of Fluids and Structures 104 (2021).
# https://doi.org/10.1016/j.jfluidstructs.2021.103312

include("../validation_util.jl")
using TrixiParticles
using OrdinaryDiffEqSymplecticRK
using JSON

tspan = (0, 10)

# `n_particles_beam_y = 5` means that the beam is 5 particles thick.
# This number is used to set the resolution of the simulation.
# The reference files are available for 5, 9, 17, 33, 65, 129, 257, 513.
# The reference files for 129, 257, 513 only cover the first swing (`tspan = (0, 0.6)`).
n_particles_beam_y = 5

# ==========================================================================================
# ==== Experiment Setup
elastic_beam = (length=0.35, thickness=0.02)
cylinder_radius = 0.05
density = 1000.0

# The structure starts at the position of the first particle and ends
# at the position of the last particle.
particle_spacing = elastic_beam.thickness / (n_particles_beam_y - 1)

# Add particle_spacing/2 to the cylinder_radius to ensure that particles are also placed on the radius
cylinder = SphereShape(particle_spacing, cylinder_radius + particle_spacing / 2,
                       (0.0, elastic_beam.thickness / 2), density,
                       cutout_min=(0.0, 0.0),
                       cutout_max=(cylinder_radius, elastic_beam.thickness),
                       place_on_shell=true, coordinates_eltype=Float64)

n_particles_clamp_x = round(Int, cylinder_radius / particle_spacing)

# Beam and clamped particles.
n_particles_per_dimension = (round(Int, elastic_beam.length / particle_spacing) +
                             n_particles_clamp_x + 1, n_particles_beam_y)

# The setup of O'Connor and Rogers places the particles on the shell of the beam.
# Note that this makes the discrete beam one particle spacing thicker than the actual beam.
beam = RectangularShape(particle_spacing, n_particles_per_dimension,
                        (0.0, 0.0); density, place_on_shell=true,
                        coordinates_eltype=Float64)

# Only the cylinder particles are clamped. The part of the beam inside the cylinder is not.
structure = union(cylinder, beam)
clamped_particles = 1:nparticles(cylinder)

# Use the example for the remaining setup, but with the geometry defined above.
# Overwrite `sol` assignment to skip time integration.
parallelization_backend = PolyesterBackend()
trixi_include(@__MODULE__, joinpath(examples_dir(), "structure", "oscillating_beam_2d.jl");
              particle_spacing, structure, clamped_particles, tspan,
              parallelization_backend, sol=nothing)

# Track the deflection of the center of the tip of the beam.
# Tracking individual particles is not the correct approach, because particles are
# interpolation points, so the motion is reconstructed by an SPH interpolation over
# the surrounding particles. Note that `Semidiscretization` creates a deep copy of the
# structure system, which means we have to extract the new system from `semi`.
tip_position = (cylinder_radius + elastic_beam.length, elastic_beam.thickness / 2)
deflection_x = StructureMotionCalculator(semi.systems[1], semi, tip_position,
                                         quantity=motion -> motion.displacement[1])
deflection_y = StructureMotionCalculator(semi.systems[1], semi, tip_position,
                                         quantity=motion -> motion.displacement[2])

pp_callback = PostprocessCallback(; deflection_x, deflection_y, dt=0.01,
                                  output_directory="out",
                                  filename="validation_run_oscillating_beam_2d_oconnor_$n_particles_beam_y",
                                  write_csv=false, write_file_interval=10)
info_callback = InfoCallback(interval=10000)

callbacks = CallbackSet(info_callback, pp_callback, stepsize_callback)

# `dt` is overwritten by the `StepsizeCallback`.
sol = solve(ode, VelocityVerlet(), dt=1.0, maxiters=10^8,
            save_everystep=false, callback=callbacks)

# Compare against the reference results generated with TrixiParticles.jl,
# NOT the results from O'Connor & Rogers (2021), which are used in the plots.
reference_file_name = joinpath(validation_dir(), "oscillating_beam_2d",
                               "validation_reference_oconnor_$n_particles_beam_y.json")
run_file_name = joinpath("out",
                         "validation_run_oscillating_beam_2d_oconnor_$n_particles_beam_y.json")

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

println("Validation results for oscillating beam 2D (O'Connor setup) with $n_particles_beam_y particles in beam thickness:")
println("  MSE deflection x: $error_deflection_x")
println("  MSE deflection y: $error_deflection_y")

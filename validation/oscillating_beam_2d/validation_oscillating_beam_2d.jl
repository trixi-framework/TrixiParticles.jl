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
using OrdinaryDiffEqLowStorageRK
using JSON

tspan = (0, 10)

# `n_particles_beam_y = 5` means that the beam is 5 particles thick.
# This number is used to set the resolution of the simulation.
# It has to be odd, so that a particle is exactly in the middle of the tip of the beam.
# Use 5, 9, 21, 35 for validation.
# Note: 35 takes a very long time!
n_particles_beam_y = 5

# Overwrite `sol` assignment to skip time integration
trixi_include(@__MODULE__, joinpath(examples_dir(), "structure", "oscillating_beam_2d.jl");
              n_particles_y=n_particles_beam_y, sol=nothing, tspan,
              penalty_force=PenaltyForceGanzenmueller(alpha=0.01))

# The reference data tracks the position of the particle in the middle of the tip
# of the beam, so we track the same quantity here for the comparison below.
middle_particle_id = Int(n_particles_per_dimension[1] * (n_particles_per_dimension[2] + 1) /
                         2)

# Make these constants because global variables in the functions below are slow
const STARTPOSITION_X = beam.coordinates[1, middle_particle_id]
const STARTPOSITION_Y = beam.coordinates[2, middle_particle_id]

function deflection_x(system, data, t)
    return data.coordinates[1, middle_particle_id] - STARTPOSITION_X
end

function deflection_y(system, data, t)
    return data.coordinates[2, middle_particle_id] - STARTPOSITION_Y
end

# Additionally track the deflection with a `StructureMotionCalculator`, which reconstructs
# the motion around the center of the tip by an SPH interpolation over the surrounding
# particles. This is less sensitive to the resolution than tracking a single particle,
# but it cannot be compared to the reference data, which uses the particle position.
tip_position = (clamp_radius + elastic_beam.length, elastic_beam.thickness / 2)

# Note that `Semidiscretization` creates a deep copy of the structure system,
# which means we have to extract the new system from `semi`.
structure_system_new = semi.systems[1]

deflection_x_interpolated = StructureMotionCalculator(structure_system_new, semi,
                                                      tip_position,
                                                      quantity=motion -> motion.displacement[1])
deflection_y_interpolated = StructureMotionCalculator(structure_system_new, semi,
                                                      tip_position,
                                                      quantity=motion -> motion.displacement[2])

pp_callback = PostprocessCallback(; deflection_x, deflection_y,
                                  deflection_x_interpolated, deflection_y_interpolated,
                                  dt=0.01, output_directory="out",
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

println("Validation results for oscillating beam 2D with $n_particles_beam_y particles in beam thickness:")
println("  MSE deflection x: $error_deflection_x")
println("  MSE deflection y: $error_deflection_y")

# This file computes the pressure sensor data of the dam break setup described in
#
# J. J. De Courcy, T. C. S. Rendall, L. Constantin, B. Titurus, J. E. Cooper.
# "Incompressible δ-SPH via artificial compressibility".
# In: Computer Methods in Applied Mechanics and Engineering, Volume 420 (2024),
# https://doi.org/10.1016/j.cma.2023.116700

include("../validation_util.jl")

using TrixiParticles
using TrixiParticles.JSON

# When using data center CPUs with large numbers of cores, especially on multi-socket
# systems with multiple NUMA nodes, pinning threads to cores can significantly
# improve performance, even for low resolutions.
# using ThreadPinning
# pinthreads(:numa)

# `resolution` in this case is set relative to `H`, the initial height of the fluid.
# Use 40, 80 or 400 for validation.
# Note: 400 takes about 30 minutes on a large data center CPU (much longer with serial update)
resolution = 40

# Use `SerialUpdate()` to obtain consistent results when using multiple threads
update_strategy = nothing
# update_strategy = SerialUpdate()

# ==========================================================================================
# ==== WCSPH simulation
trixi_include(@__MODULE__,
              joinpath(validation_dir(), "dam_break_2d",
                       "setup_marrone_2011.jl"),
              use_edac=false, update_strategy=update_strategy,
              particles_per_height=resolution,
              sound_speed=50 * sqrt(9.81 * 0.6), # This is used by De Courcy et al. (2024)
              alpha=0.01, # This is used by De Courcy et al. (2024)
              tspan=(0.0, 7 / sqrt(9.81 / 0.6))) # This is used by De Courcy et al. (2024)

reference_file_wcsph_name = joinpath(validation_dir(), "dam_break_2d",
                                     "validation_reference_wcsph_$formatted_string.json")
run_file_wcsph_name = joinpath("out",
                               "validation_result_dam_break_wcsph_$formatted_string.json")

reference_data = JSON.parsefile(reference_file_wcsph_name)
run_data = JSON.parsefile(run_file_wcsph_name)

error_wcsph_P1 = interpolated_mse(reference_data["pressure_P1_fluid_1"]["time"],
                                  reference_data["pressure_P1_fluid_1"]["values"],
                                  run_data["pressure_P1_fluid_1"]["time"],
                                  run_data["pressure_P1_fluid_1"]["values"])

error_wcsph_P2 = interpolated_mse(reference_data["pressure_P2_fluid_1"]["time"],
                                  reference_data["pressure_P2_fluid_1"]["values"],
                                  run_data["pressure_P2_fluid_1"]["time"],
                                  run_data["pressure_P2_fluid_1"]["values"])

# ==========================================================================================
# ==== EDAC simulation
trixi_include(@__MODULE__,
              joinpath(validation_dir(), "dam_break_2d",
                       "setup_marrone_2011.jl"),
              use_edac=true, update_strategy=update_strategy,
              particles_per_height=resolution,
              sound_speed=50 * sqrt(9.81 * 0.6), # This is used by De Courcy et al. (2024)
              alpha=0.01, # This is used by De Courcy et al. (2024)
              tspan=(0.0, 7 / sqrt(9.81 / 0.6))) # This is used by De Courcy et al. (2024)

reference_file_edac_name = joinpath(validation_dir(), "dam_break_2d",
                                    "validation_reference_edac_$formatted_string.json")
run_file_edac_name = joinpath("out",
                              "validation_result_dam_break_edac_$formatted_string.json")

reference_data = JSON.parsefile(reference_file_edac_name)
run_data = JSON.parsefile(run_file_edac_name)

error_edac_P1 = interpolated_mse(reference_data["pressure_P1_fluid_1"]["time"],
                                 reference_data["pressure_P1_fluid_1"]["values"],
                                 run_data["pressure_P1_fluid_1"]["time"],
                                 run_data["pressure_P1_fluid_1"]["values"])

error_edac_P2 = interpolated_mse(reference_data["pressure_P2_fluid_1"]["time"],
                                 reference_data["pressure_P2_fluid_1"]["values"],
                                 run_data["pressure_P2_fluid_1"]["time"],
                                 run_data["pressure_P2_fluid_1"]["values"])

print(error_edac_P1, " ", error_edac_P2, " ",
      error_wcsph_P1, " ", error_wcsph_P2, "\n")

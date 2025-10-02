# This file runs a validation based on the dam break setup described in
#
# S. Marrone, M. Antuono, A. Colagrossi, G. Colicchio, D. le Touzé, G. Graziani.
# "δ-SPH model for simulating violent impact flows".
# In: Computer Methods in Applied Mechanics and Engineering, Volume 200, Issues 13–16 (2011), pages 1526–1542.
# https://doi.org/10.1016/J.CMA.2010.12.016

include("../validation_util.jl")

using TrixiParticles
using JSON

# `resolution` in this case is set relative to `H`, the initial height of the fluid.
# Use 40, 80 or 400 for validation.
# Note: 400 takes a few hours!
resolution = 40

# Use `SerialUpdate()` to obtain consistent results across different numbers of threads
update_strategy = nothing
update_strategy = SerialUpdate()

# ==========================================================================================
# ==== WCSPH simulation
trixi_include(@__MODULE__,
              joinpath(validation_dir(), "dam_break_2d",
                       "setup_marrone_2011.jl"),
              use_edac=false, update_strategy=update_strategy,
              particles_per_height=resolution)

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
              particles_per_height=resolution)

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

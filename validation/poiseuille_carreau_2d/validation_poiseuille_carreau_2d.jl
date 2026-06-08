using TrixiParticles

# ==========================================================================================
# 2D Periodic Poiseuille Flow Validation for Carreau-Yasuda Fluids
#
# This validation runs the Carreau-Yasuda Poiseuille example for several
# power-law indices and checks the final relative L2 velocity error against the
# analytical steady profile.
# ==========================================================================================

# ==========================================================================================
# ==== Resolution
# The default resolution is intentionally modest so this validation is practical to run
# during review. For the high-resolution run inspired by Coclite et al. (2020), use
# `ny = 200`.
ny = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 50

# ==========================================================================================
# ==== Experiment Setup
# Optional command line arguments:
#   1. ny
#   2. t_end_factor
#   3. eps_factor
#   4. sound_speed_factor
#   5. initial condition mode: "newtonian", "analytical", or "zero"
#   6. comma-separated Carreau-Yasuda n values
default_n_values = (1.0, 1.5, 0.5, 0.25)
n_values = length(ARGS) >= 6 ? Tuple(parse.(Float64, split(ARGS[6], ','))) :
           default_n_values

t_end_factor = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.1
eps_factor = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1.0
sound_speed_factor = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 60.0
initial_condition_mode = length(ARGS) >= 5 ? lowercase(strip(ARGS[5])) : "analytical"

# These bounds are intentionally looser than the observed high-resolution values
# so that the validation checks the expected behaviour without becoming brittle
# across machines or small time-step differences.
relative_l2_error_bounds = Dict(0.25 => 0.06,
                                0.5 => 0.06,
                                1.0 => 0.06,
                                1.5 => 0.06)

final_relative_l2_errors = Dict{Float64, Float64}()
final_max_velocity_errors = Dict{Float64, Float64}()

function final_error_values(output_directory, result_filename)
    csv_file = joinpath(output_directory, result_filename * ".csv")
    data = TrixiParticles.CSV.read(csv_file, TrixiParticles.DataFrame)
    column_names = string.(names(data))
    l2_column = Symbol(only(filter(name -> startswith(name, "l2_velocity_error"),
                                   column_names)))
    max_column = Symbol(only(filter(name -> startswith(name, "max_velocity_error"),
                                    column_names)))
    return data[!, l2_column][end], data[!, max_column][end]
end

# ==========================================================================================
# ==== Run Simulations
for power_law_index in n_values
    println("\n--- Running Carreau-Yasuda Poiseuille validation with n = ",
            power_law_index, " ---")

    trixi_include(@__MODULE__,
                  joinpath(examples_dir(), "fluid", "poiseuille_carreau_2d.jl");
                  ny, t_end_factor, eps_factor, sound_speed_factor,
                  initial_condition_mode, power_law_index)

    n_label = replace(string(power_law_index), "." => "p")
    output_directory = joinpath("out_poiseuille_carreau", "n_$power_law_index")
    result_filename = "validation_run_poiseuille_carreau_2d_n_$(n_label)_ny_$ny"

    relative_l2_error,
    max_velocity_error = final_error_values(output_directory,
                                            result_filename)
    final_relative_l2_errors[power_law_index] = relative_l2_error
    final_max_velocity_errors[power_law_index] = max_velocity_error

    @assert relative_l2_error <= relative_l2_error_bounds[power_law_index] "relative L2 error $(relative_l2_error) exceeded bound $(relative_l2_error_bounds[power_law_index]) for n = $(power_law_index)"
end

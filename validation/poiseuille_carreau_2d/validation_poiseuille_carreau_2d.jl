using TrixiParticles

include("analytical_solution.jl")

# ==========================================================================================
# 2D Periodic Poiseuille Flow Validation for Carreau-Yasuda Fluids
#
# This validation runs the Carreau-Yasuda Poiseuille example for several
# power-law indices and checks the final relative L2 velocity error against the
# analytical steady profile.
# ==========================================================================================

# ==========================================================================================
# ==== Resolution
# The default resolution is intentionally modest so this validation is practical to run.
# Override it interactively via `trixi_include(...; ny=...)`.
ny = 50

# ==========================================================================================
# ==== Experiment Setup
default_n_values = (1.0, 1.5, 0.5, 0.25)
n_values = default_n_values

t_end_factor = 0.1
eps_factor = 1.0
sound_speed_factor = 60.0
initial_condition_mode = :analytical
parallelization_backend = PolyesterBackend()

channel_height = 1.0
channel_length = 6.0 * channel_height
fluid_density = 1000.0
nu0 = 1.0e-3
nu_inf = 0.0
lambda_exponent = 2.0
reynolds_number = 200.0

reference_velocity = reynolds_number * nu0 / channel_height
pressure_gradient = 8.0 * fluid_density * reference_velocity^2 /
                    (reynolds_number * channel_height)
carreau_time_constant = channel_height / reference_velocity

# These bounds are intentionally looser than the observed high-resolution values
# so that the validation checks the expected behaviour without becoming brittle
# across machines or small time-step differences.
relative_l2_error_bounds = Dict(0.25 => 0.06,
                                0.5 => 0.06,
                                1.0 => 0.06,
                                1.5 => 0.06)

final_relative_l2_errors = Dict{Float64, Float64}()
final_max_velocity_errors = Dict{Float64, Float64}()

mean_velocity_x(system, data, t) = nothing
function mean_velocity_x(system::TrixiParticles.AbstractFluidSystem, data, t)
    return sum(@view data.velocity[1, :]) / size(data.velocity, 2)
end

interpolated_velocity_profile(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = nothing

function interpolated_velocity_profile(system::TrixiParticles.AbstractFluidSystem,
                                       dv_ode, du_ode, v_ode, u_ode, semi, t)
    interpolation_result = interpolate_line([0.5 * channel_length, 0.0],
                                            [0.5 * channel_length, channel_height],
                                            ny + 1, semi, system, v_ode, u_ode;
                                            endpoint=true, cut_off_bnd=false)

    return collect(stack(interpolation_result.velocity)[1, :])
end

function profile_history(output_directory, result_filename)
    json_file = joinpath(output_directory, result_filename * ".json")
    data = TrixiParticles.JSON.parsefile(json_file; allownan=true)
    profile_key = only(filter(name -> startswith(name, "interpolated_velocity_profile"),
                              collect(keys(data))))
    times = Float64.(data[profile_key]["time"])
    profiles = [replace!(Float64.(profile), NaN => 0.0)
                for profile in data[profile_key]["values"]]

    return times, profiles
end

function error_history(profiles, power_law_index)
    y_positions = collect(range(0.0, channel_height; length=length(first(profiles))))
    analytical_velocity = analytical_ux_profile(y_positions, power_law_index,
                                                channel_height, fluid_density, nu0,
                                                nu_inf, carreau_time_constant,
                                                lambda_exponent, pressure_gradient)

    relative_l2_errors = Float64[]
    max_velocity_errors = Float64[]

    for profile in profiles
        relative_l2_error, max_velocity_error = velocity_profile_errors(profile,
                                                                        analytical_velocity)
        push!(relative_l2_errors, relative_l2_error)
        push!(max_velocity_errors, max_velocity_error)
    end

    return relative_l2_errors, max_velocity_errors
end

# ==========================================================================================
# ==== Run Simulations
for power_law_index in n_values
    println("\n--- Running Carreau-Yasuda Poiseuille validation with n = ",
            power_law_index, " ---")

    n_label = replace(string(power_law_index), "." => "p")
    output_directory = joinpath("out_poiseuille_carreau", "n_$power_law_index")
    result_filename = "validation_run_poiseuille_carreau_2d_n_$(n_label)_ny_$ny"

    pp_callback = PostprocessCallback(; dt=t_end_factor * channel_height /
                                           reference_velocity / 20,
                                      output_directory,
                                      filename=result_filename,
                                      mean_velocity_x,
                                      interpolated_velocity_profile,
                                      write_csv=false,
                                      write_file_interval=0)

    trixi_include(@__MODULE__,
                  joinpath(examples_dir(), "fluid", "poiseuille_carreau_2d.jl");
                  ny, t_end_factor, eps_factor, sound_speed_factor,
                  initial_condition_mode=QuoteNode(initial_condition_mode),
                  power_law_index, parallelization_backend, pp_callback)

    _, profiles = profile_history(output_directory, result_filename)
    relative_l2_errors, max_velocity_errors = error_history(profiles, power_law_index)

    relative_l2_error = last(relative_l2_errors)
    max_velocity_error = last(max_velocity_errors)
    final_relative_l2_errors[power_law_index] = relative_l2_error
    final_max_velocity_errors[power_law_index] = max_velocity_error

    @assert relative_l2_error <= relative_l2_error_bounds[power_law_index] "relative L2 error $(relative_l2_error) exceeded bound $(relative_l2_error_bounds[power_law_index]) for n = $(power_law_index)"
end

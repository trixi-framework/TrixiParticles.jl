using TrixiParticles
using OrdinaryDiffEqLowStorageRK
using Dates

include("validation_util.jl")

# ==========================================================================================
# 2D Periodic Poiseuille Flow Validation for Carreau-Yasuda Fluids
#
# This validation runs the Carreau-Yasuda Poiseuille example for several
# power-law indices and checks the final relative L2 velocity error against the
# analytical steady profile.
# ==========================================================================================

# Load the example's setup and analytical helpers without assembling or solving the ODE.
example_file = joinpath(examples_dir(), "fluid", "poiseuille_carreau_2d.jl")
trixi_include(@__MODULE__, example_file; ode=nothing, sol=nothing,
              ny=50, t_end_factor=0.1,
              initial_condition_mode=:analytical, viscosity_model=:carreau,
              smoothing_kernel=WendlandC2Kernel{2}())

# Maximum accepted relative L2 error for each power-law index.
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
                                            endpoint=true, cut_off_bnd=false,
                                            include_wall_velocity=true)

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
        relative_l2_error,
        max_velocity_error = velocity_profile_errors(profile,
                                                     analytical_velocity)
        push!(relative_l2_errors, relative_l2_error)
        push!(max_velocity_errors, max_velocity_error)
    end

    return relative_l2_errors, max_velocity_errors
end

# ==========================================================================================
# ==== Run Simulations
n_values = (1.0, 1.5, 0.5, 0.25)
output_root = joinpath("out_poiseuille_carreau",
                       "run_" * Dates.format(now(), dateformat"yyyymmdd_HHMMSS"))
check_error_bounds = initial_condition_mode == :analytical

for power_law_index in n_values
    println("\n--- Running Carreau-Yasuda Poiseuille validation with n = ",
            power_law_index, " ---")

    n_label = replace(string(power_law_index), "." => "p")
    local output_directory = joinpath(output_root, "n_$power_law_index")
    result_filename = "validation_run_poiseuille_carreau_2d_n_$(n_label)_ny_$ny"

    local pp_callback = PostprocessCallback(;
                                            dt=t_end_factor * channel_height /
                                               reference_velocity / 20,
                                            output_directory,
                                            filename=result_filename,
                                            mean_velocity_x,
                                            interpolated_velocity_profile,
                                            write_csv=false,
                                            write_file_interval=0)

    trixi_include(@__MODULE__, example_file;
                  ny, t_end_factor, power_law_index, pp_callback, output_directory,
                  smoothing_kernel=WendlandC2Kernel{2}(),
                  initial_condition_mode, viscosity_model)

    _, profiles = profile_history(output_directory, result_filename)
    relative_l2_errors, max_velocity_errors = error_history(profiles, power_law_index)

    relative_l2_error = last(relative_l2_errors)
    max_velocity_error = last(max_velocity_errors)
    final_relative_l2_errors[power_law_index] = relative_l2_error
    final_max_velocity_errors[power_law_index] = max_velocity_error

    if check_error_bounds
        @assert relative_l2_error <= relative_l2_error_bounds[power_law_index] "relative L2 error $(relative_l2_error) exceeded bound $(relative_l2_error_bounds[power_law_index]) for n = $(power_law_index)"
    else
        @info "Skipping steady-profile error bound for transient initial condition" power_law_index initial_condition_mode relative_l2_error
    end
end

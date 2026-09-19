#!/usr/bin/env julia
using Glob, Plots
using TrixiParticles
using TrixiParticles.JSON

include("analytical_solution.jl")

gr()

function profile_history(json_file::AbstractString)
    data = JSON.parsefile(json_file; allownan=true)
    profile_key = only(filter(name -> startswith(name, "interpolated_velocity_profile"),
                              collect(keys(data))))
    times = Float64.(data[profile_key]["time"])
    profiles = [replace!(Float64.(profile), NaN => 0.0)
                for profile in data[profile_key]["values"]]

    return times, profiles
end

function plot_carreau(out_root::AbstractString="out_poiseuille_carreau")
    n_dirs = sort(glob(joinpath(out_root, "n_*")))
    isempty(n_dirs) && error("No n_* directories found under $out_root")

    channel_height = 1.0
    fluid_density = 1000.0
    nu0 = 1.0e-3
    nu_inf = 0.0
    lambda_exponent = 2.0
    reynolds_number = 200.0
    reference_velocity = reynolds_number * nu0 / channel_height
    pressure_gradient = 8.0 * fluid_density * reference_velocity^2 /
                        (reynolds_number * channel_height)
    carreau_time_constant = channel_height / reference_velocity

    profile_plot = plot(title="Final profile",
                        xlabel="y / H",
                        ylabel="u_x / u_max",
                        legend=:bottomright,
                        xlim=(0.0, 1.0),
                        ylim=(0.0, 1.05),
                        size=(600, 600))
    error_plot = plot(title="Relative L2 error vs analytical profile",
                      xlabel="time",
                      ylabel="relative L2 error",
                      legend=:bottomright,
                      size=(600, 450))

    for n_dir in n_dirs
        json_files = sort(glob("validation_run_poiseuille_carreau_2d_*.json", n_dir))
        if isempty(json_files)
            @warn "Missing validation JSON file for $n_dir, skipping"
            continue
        end

        json_file = last(json_files)
        n_val = parse(Float64, replace(basename(n_dir), "n_" => ""))
        times, profiles = profile_history(json_file)
        y_positions = collect(range(0.0, channel_height; length=length(first(profiles))))
        analytical_velocity = analytical_ux_profile(y_positions, n_val, channel_height,
                                                    fluid_density, nu0, nu_inf,
                                                    carreau_time_constant,
                                                    lambda_exponent,
                                                    pressure_gradient)

        relative_l2_errors = Float64[]
        for profile in profiles
            relative_l2_error, _ = velocity_profile_errors(profile, analytical_velocity)
            push!(relative_l2_errors, relative_l2_error)
        end

        final_profile = last(profiles)
        u_max = maximum(analytical_velocity)
        y_norm = y_positions ./ channel_height
        u_norm = final_profile ./ u_max
        analytical_norm = analytical_velocity ./ u_max

        plot!(profile_plot, y_norm, u_norm;
              label="n=$(n_val)", linewidth=2, linestyle=:solid)
        plot!(profile_plot, y_norm, analytical_norm;
              label="n=$(n_val) analytical", linewidth=2, linestyle=:dash)

        per_case_fig = plot(layout=(@layout [a{0.55h}; b{0.45h}]), size=(900, 800))
        plot!(per_case_fig[1], y_norm, u_norm;
              label="numerical", linewidth=2, linestyle=:solid, color=:blue)
        plot!(per_case_fig[1], y_norm, analytical_norm;
              label="analytical", linestyle=:dash, linewidth=2, color=:black)
        xlabel!(per_case_fig[1], "y / H")
        ylabel!(per_case_fig[1], "u_x / u_max")
        title!(per_case_fig[1], "n=$(n_val) final profile")
        xlims!(per_case_fig[1], (0.0, 1.0))
        ylims!(per_case_fig[1], (0.0, 1.05))

        plot!(per_case_fig[2], times, relative_l2_errors;
              label="relative L2", linewidth=2, color=:black)
        xlabel!(per_case_fig[2], "time")
        ylabel!(per_case_fig[2], "relative L2 error")
        title!(per_case_fig[2], "n=$(n_val) relative error vs time")

        plot!(error_plot, times, relative_l2_errors;
              label="n=$(n_val)", linewidth=2)

        savefig(per_case_fig, joinpath(n_dir, "carreau_comparison_and_error.png"))
    end

    savefig(profile_plot, joinpath(out_root, "carreau_final_profiles.png"))
    savefig(error_plot, joinpath(out_root, "carreau_error_comparison.png"))
    println("Saved profile comparison to: ",
            joinpath(out_root, "carreau_final_profiles.png"))
    println("Saved error comparison to: ",
            joinpath(out_root, "carreau_error_comparison.png"))
end

plot_carreau(length(ARGS) > 0 ? ARGS[1] : "out_poiseuille_carreau")

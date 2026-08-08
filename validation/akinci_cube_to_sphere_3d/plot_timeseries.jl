using CairoMakie
using CSV
using DataFrames
using TrixiParticles

case_directory = joinpath(validation_dir(), "akinci_cube_to_sphere_3d")
summary_input_file = get(ARGS, 1,
                         joinpath(case_directory, "out", "akinci_cube_to_sphere_3d.csv"))
timeseries_input_file = get(ARGS, 2,
                            joinpath(case_directory, "out",
                                     "akinci_cube_to_sphere_3d_timeseries.csv"))
asphericity_output_file = get(ARGS, 3,
                              joinpath(case_directory, "out",
                                       "akinci_cube_to_sphere_3d_asphericity.png"))
particle_quality_output_file = get(ARGS, 4,
                                   joinpath(case_directory, "out",
                                            "akinci_cube_to_sphere_3d_particle_quality.png"))

summary = CSV.read(summary_input_file, DataFrame)
time_series = CSV.read(timeseries_input_file, DataFrame)

method_order = [("wcsph", "ContinuityDensity"),
    ("wcsph", "SummationDensity"),
    ("edac", "ContinuityDensity"),
    ("edac", "SummationDensity"),
    ("iisph", "SummationDensity")]
method_names = Dict(("wcsph", "ContinuityDensity") => "WCSPH + Continuity",
                    ("wcsph", "SummationDensity") => "WCSPH + Summation",
                    ("edac", "ContinuityDensity") => "EDAC + Continuity",
                    ("edac", "SummationDensity") => "EDAC + Summation",
                    ("iisph", "SummationDensity") => "IISPH + Summation")
method_colors = [:dodgerblue3, :seagreen4, :darkorange2, :mediumpurple3, :firebrick3]
method_linestyles = [:solid, :dash, :dot, :dashdot, :solid]

function winner_row(summary, method)
    rows = summary[(summary.sph_method .== method[1]) .& (summary.density_calculator .== method[2]),
                   :]
    isempty(rows) && error("missing summary rows for $(method[1]) + $(method[2])")
    return rows[argmin(rows.late_time_mean_asphericity), :]
end

function variant_name(row)
    if row.sph_method == "wcsph"
        return row.correction == "none" ? "no correction" : "Akinci correction"
    elseif row.sph_method == "edac"
        pressure = row.pressure_formulation == "density_matched" ?
                   "density-matched pressure" : "inter-particle pressure"
        return row.correction == "none" ? pressure : pressure * " + Akinci correction"
    elseif row.sph_method == "iisph" && row.correction != "none"
        return "implicit pressure + Akinci correction"
    end
    return "implicit pressure"
end

winners = [winner_row(summary, method) for method in method_order]

function plot_history(metric, title, y_label, output_file)
    figure = Figure(size=(1900, 1050), backgroundcolor=:white)
    Label(figure[0, 1], title; fontsize=34, font=:bold, padding=(0, 0, 6, 0))
    Label(figure[1, 1],
          "Lowest mean asphericity over the final 20% for each formulation + density calculator";
          fontsize=18, color=:gray30, padding=(0, 0, 14, 0))
    axis = Axis(figure[2, 1]; xlabel="time / s", ylabel=y_label,
                xlabelsize=20, ylabelsize=20, xticklabelsize=16, yticklabelsize=16,
                xgridcolor=(:gray70, 0.35), ygridcolor=(:gray70, 0.35))

    for (index, (method, winner)) in enumerate(zip(method_order, winners))
        history = time_series[(time_series.case .== winner.case) .& (time_series.model .== winner.model),
                              :]
        isempty(history) &&
            error("missing time series for $(winner.model) / $(winner.case)")
        sort!(history, :time)
        label = method_names[method] * " | " * variant_name(winner)
        lines!(axis, history.time, history[!, metric]; color=method_colors[index],
               linestyle=method_linestyles[index], linewidth=4, label)
        scatter!(axis, history.time[end:end], history[end:end, metric];
                 color=method_colors[index], markersize=14)
    end

    xlims!(axis, minimum(time_series.time), maximum(time_series.time))
    ylims!(axis, 0, nothing)
    axislegend(axis; position=:rt, labelsize=16, framevisible=true,
               backgroundcolor=(:white, 0.9))
    Label(figure[3, 1],
          "Winner selection is time-averaged to avoid ranking oscillating drops at one snapshot; " *
          "lower values are better.";
          fontsize=16, color=:gray35, padding=(0, 0, 0, 8))

    mkpath(dirname(output_file))
    save(output_file, figure; px_per_unit=1.5)
    println("Saved Akinci time-history graphic to $output_file")
    return figure
end

asphericity_figure = plot_history(:asphericity,
                                  "Asphericity over time",
                                  "support-radius CV (0 = sphere)",
                                  asphericity_output_file)
particle_quality_figure = plot_history(:particle_spacing_cv,
                                       "Particle distribution quality over time",
                                       "nearest-neighbor distance CV (0 = uniform)",
                                       particle_quality_output_file)

(; asphericity_figure, particle_quality_figure)

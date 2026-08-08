using CairoMakie
using CSV
using DataFrames
using TrixiParticles

case_directory = joinpath(validation_dir(), "akinci_cube_to_sphere_3d")
input_file = get(ARGS, 1,
                 joinpath(case_directory, "out",
                          "akinci_cube_to_sphere_3d_resolution.csv"))
output_file = get(ARGS, 2,
                  joinpath(case_directory, "out",
                           "akinci_cube_to_sphere_3d_resolution.png"))

results = CSV.read(input_file, DataFrame)

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
method_markers = [:circle, :rect, :diamond, :utriangle, :cross]

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

resolution_levels = sort(unique(results.particles_per_dimension))
expected_rows = length(method_order) * length(resolution_levels)
nrow(results) == expected_rows ||
    error("expected $expected_rows resolution rows in '$input_file', found $(nrow(results))")
resolution_labels = ["$(resolution)^3\n($(resolution^3) particles)"
                     for resolution in resolution_levels]

figure = Figure(size=(2100, 1180), backgroundcolor=:white)
Label(figure[0, 1:2], "Resolution sensitivity of the best Akinci configurations";
      fontsize=34, font=:bold, padding=(0, 0, 6, 0))
Label(figure[1, 1:2],
      "Winners selected by late-time mean asphericity at the baseline resolution; " *
      "the fixed empirical coefficient is not recalibrated across resolutions";
      fontsize=18, color=:gray30, padding=(0, 0, 14, 0))

axis_asphericity = Axis(figure[2, 1]; title="Late-time mean asphericity",
                        xlabel="particles per initial cube edge",
                        ylabel="support-radius CV (0 = sphere)",
                        xticks=(resolution_levels, resolution_labels),
                        titlesize=24, xlabelsize=19, ylabelsize=19,
                        xticklabelsize=16, yticklabelsize=16)
axis_quality = Axis(figure[2, 2]; title="Late-time mean particle distribution quality",
                    xlabel="particles per initial cube edge",
                    ylabel="nearest-neighbor distance CV (0 = uniform)",
                    xticks=(resolution_levels, resolution_labels),
                    titlesize=24, xlabelsize=19, ylabelsize=19,
                    xticklabelsize=16, yticklabelsize=16)

for (index, method) in enumerate(method_order)
    method_results = results[(results.sph_method .== method[1]) .& (results.density_calculator .== method[2]),
                             :]
    nrow(method_results) == length(resolution_levels) ||
        error("incomplete resolution data for $(method[1]) + $(method[2])")
    sort!(method_results, :particles_per_dimension)
    label = method_names[method] * " | " * variant_name(method_results[1, :])

    lines!(axis_asphericity, method_results.particles_per_dimension,
           method_results.late_time_mean_asphericity;
           color=method_colors[index], linewidth=4, label)
    scatter!(axis_asphericity, method_results.particles_per_dimension,
             method_results.late_time_mean_asphericity;
             color=method_colors[index], marker=method_markers[index], markersize=18)
    lines!(axis_quality, method_results.particles_per_dimension,
           method_results.late_time_mean_particle_spacing_cv;
           color=method_colors[index], linewidth=4)
    scatter!(axis_quality, method_results.particles_per_dimension,
             method_results.late_time_mean_particle_spacing_cv;
             color=method_colors[index], marker=method_markers[index], markersize=18)
end

ylims!(axis_asphericity, 0, nothing)
ylims!(axis_quality, 0, nothing)
axislegend(axis_asphericity; position=:rt, labelsize=14, framevisible=true,
           backgroundcolor=(:white, 0.9))
Label(figure[3, 1:2],
      "Final time: $(only(unique(results.final_time))) s | " *
      "constant acoustic CFL: $(only(unique(results.acoustic_cfl))) | " *
      "h/delta: $(only(unique(results.smoothing_length_factor)))";
      fontsize=16, color=:gray35, padding=(0, 0, 0, 8))
Label(figure[4, 1:2],
      "The same-material normal term uses dimensionless local normals and harmonic pair mass; " *
      "no physical convergence is claimed.";
      fontsize=17, color=:gray25, font=:bold, padding=(0, 0, 0, 10))
colgap!(figure.layout, 32)
mkpath(dirname(output_file))
save(output_file, figure; px_per_unit=1.5)
println("Saved Akinci resolution graphic to $output_file")

figure

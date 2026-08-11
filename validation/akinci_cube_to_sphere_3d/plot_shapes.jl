using CairoMakie
using CSV
using DataFrames
using Printf
using Statistics: mean
using TrixiParticles

case_directory = joinpath(validation_dir(), "akinci_cube_to_sphere_3d")
shape_input_file = get(ARGS, 1,
                       joinpath(case_directory, "out",
                                "akinci_cube_to_sphere_3d_shapes.csv"))
metric_input_file = get(ARGS, 2,
                        joinpath(case_directory, "out", "akinci_cube_to_sphere_3d.csv"))
output_file = get(ARGS, 3,
                  joinpath(case_directory, "out", "akinci_cube_to_sphere_3d_shapes.png"))

shapes = CSV.read(shape_input_file, DataFrame)
metrics = CSV.read(metric_input_file, DataFrame)

case_order = ["wcsph_continuity",
    "wcsph_continuity_akinci_free_surface_correction",
    "wcsph_summation",
    "wcsph_summation_akinci_free_surface_correction",
    "edac_continuity_inter_particle",
    "edac_continuity_inter_particle_akinci_free_surface_correction",
    "edac_continuity_density_matched",
    "edac_continuity_density_matched_akinci_free_surface_correction",
    "edac_summation_inter_particle",
    "edac_summation_inter_particle_akinci_free_surface_correction",
    "edac_summation_density_matched",
    "edac_summation_density_matched_akinci_free_surface_correction",
    "iisph_summation",
    "iisph_summation_akinci_free_surface_correction"]
case_labels = ["WCSPH\nContinuityDensity",
    "WCSPH | ContinuityDensity\n+ AkinciFreeSurfaceCorrection",
    "WCSPH\nSummationDensity",
    "WCSPH | SummationDensity\n+ AkinciFreeSurfaceCorrection",
    "EDAC | ContinuityDensity\ninter_particle_averaged_pressure",
    "EDAC | ContinuityDensity | inter_particle_averaged_pressure\n+ AkinciFreeSurfaceCorrection",
    "EDAC | ContinuityDensity\ndensity-matched pressure",
    "EDAC | ContinuityDensity | density-matched pressure\n+ AkinciFreeSurfaceCorrection",
    "EDAC | SummationDensity\ninter_particle_averaged_pressure",
    "EDAC | SummationDensity | inter_particle_averaged_pressure\n+ AkinciFreeSurfaceCorrection",
    "EDAC | SummationDensity\ndensity-matched pressure",
    "EDAC | SummationDensity | density-matched pressure\n+ AkinciFreeSurfaceCorrection",
    "IISPH | SummationDensity\nimplicit pressure",
    "IISPH | SummationDensity\n+ AkinciFreeSurfaceCorrection"]
model_order = ["CohesionForceAkinci", "SurfaceTensionAkinci"]
model_labels = ["CohesionForceAkinci\ncohesion-only control",
    "SurfaceTensionAkinci\ncohesion + area minimization"]

expected_runs = length(case_order) * length(model_order)
nrow(metrics) == expected_runs ||
    error("expected $expected_runs metric rows in '$metric_input_file', found $(nrow(metrics))")

best_configurations = Set{Tuple{String, String}}()
for method_results in groupby(metrics, [:sph_method, :density_calculator])
    best_row = method_results[argmin(method_results.late_time_mean_asphericity), :]
    push!(best_configurations, (String(best_row.case), String(best_row.model)))
end

target_radius = cbrt(3 / (4 * pi))
angles = range(0, 2pi; length=181)
circle_x = target_radius .* cos.(angles)
circle_y = target_radius .* sin.(angles)
circle_zero = zeros(length(angles))
coordinate_limit = 0.78

figure = Figure(size=(1700, 4900), backgroundcolor=:white)
plot_columns = 1:(length(model_order) + 1)
Label(figure[0, plot_columns], "Final particle shapes for the Akinci model shootout";
      fontsize=36, font=:bold, padding=(0, 0, 4, 0))
Label(figure[1, plot_columns],
      "Only the SurfaceTensionAkinci column is expected to minimize surface area. " *
      "All panels share a scale; green rings show the equal-volume sphere.";
      fontsize=18, color=:gray30, padding=(0, 0, 10, 0))

for (model_index, model_label) in enumerate(model_labels)
    Label(figure[2, model_index + 1], model_label; fontsize=20, font=:bold,
          padding=(5, 5, 5, 5))
end

for (case_index, case) in enumerate(case_order)
    row = case_index + 2
    Label(figure[row, 1], case_labels[case_index]; fontsize=17, halign=:right,
          tellheight=false, padding=(5, 20, 5, 5))

    for (model_index, model) in enumerate(model_order)
        column = model_index + 1
        mask = (shapes.case .== case) .& (shapes.model .== model)
        particle_data = shapes[mask, :]
        isempty(particle_data) && error("missing shape data for $model / $case")

        metric_row = only(eachrow(metrics[(metrics.case .== case) .& (metrics.model .== model),
                                          :]))
        nrow(particle_data) == metric_row.particle_count ||
            error("expected $(metric_row.particle_count) particles for $model / $case, " *
                  "found $(nrow(particle_data))")
        support_radius_cv_change = 100 * (metric_row.final_support_radius_cv /
                                    metric_row.initial_support_radius_cv - 1)
        is_best_configuration = (case, model) in best_configurations
        is_full_model = model == "SurfaceTensionAkinci"
        score_color = is_full_model ?
                      (support_radius_cv_change < 0 ? :seagreen4 : :firebrick3) : :gray40
        score_label = is_full_model ? "rounding metric" : "control response"
        score_prefix = is_best_configuration ? "\u2713 " : ""

        side_length = only(unique(particle_data.cube_side_length))
        x = (particle_data.x .- mean(particle_data.x)) ./ side_length
        y = (particle_data.y .- mean(particle_data.y)) ./ side_length
        z = (particle_data.z .- mean(particle_data.z)) ./ side_length

        panel = GridLayout()
        figure[row, column] = panel
        Label(panel[1, 1],
              @sprintf("%s%s: %+.1f%% support CV", score_prefix, score_label,
                       support_radius_cv_change);
              fontsize=15, color=score_color, font=:bold,
              padding=(0, 0, 0, 2))
        axis = Axis3(panel[2, 1]; aspect=(1, 1, 1), perspectiveness=0.15,
                     azimuth=1.2pi, elevation=pi / 7)

        scatter!(axis, x, y, z; color=z, colorrange=(-coordinate_limit, coordinate_limit),
                 colormap=:plasma, markersize=7)
        lines!(axis, circle_x, circle_y, circle_zero;
               color=(:seagreen3, 0.7), linewidth=2)
        lines!(axis, circle_x, circle_zero, circle_y;
               color=(:seagreen3, 0.7), linewidth=2)
        lines!(axis, circle_zero, circle_x, circle_y;
               color=(:seagreen3, 0.7), linewidth=2)
        limits!(axis, -coordinate_limit, coordinate_limit,
                -coordinate_limit, coordinate_limit,
                -coordinate_limit, coordinate_limit)
        hidedecorations!(axis)
    end
end

final_time = only(unique(metrics.final_time))
particle_count = only(unique(metrics.particle_count))
Label(figure[length(case_order) + 3, plot_columns],
      "\u2713 marks the lowest mean asphericity over the final 20% of the run for each SPH " *
      "formulation + density calculator | Final time: $final_time s | " *
      "$particle_count particles per panel\n" *
      "coordinates normalized by initial cube side length | " *
      "green center-column score means movement toward a rounder result";
      fontsize=16, color=:gray30, padding=(0, 0, 0, 8))
Label(figure[length(case_order) + 4, plot_columns],
      "AkinciFreeSurfaceCorrection scales surface-tension and viscosity forces near the " *
      "free surface; it does not modify pressure.";
      fontsize=15, color=:gray40, padding=(0, 0, 0, 10))

colgap!(figure.layout, 14)
rowgap!(figure.layout, 8)
mkpath(dirname(output_file))
save(output_file, figure; px_per_unit=1.2)
println("Saved Akinci final-shape graphic to $output_file")

figure

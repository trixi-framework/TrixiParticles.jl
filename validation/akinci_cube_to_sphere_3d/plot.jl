using CairoMakie
using CSV
using DataFrames
using Printf
using TrixiParticles

case_directory = joinpath(validation_dir(), "akinci_cube_to_sphere_3d")
input_file = get(ARGS, 1,
                 joinpath(case_directory, "out", "akinci_cube_to_sphere_3d.csv"))
output_file = get(ARGS, 2,
                  joinpath(case_directory, "out", "akinci_cube_to_sphere_3d.png"))

results = CSV.read(input_file, DataFrame)

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
case_labels = ["WCSPH | ContinuityDensity",
    "WCSPH | ContinuityDensity\n+ AkinciFreeSurfaceCorrection",
    "WCSPH | SummationDensity",
    "WCSPH | SummationDensity\n+ AkinciFreeSurfaceCorrection",
    "EDAC | ContinuityDensity\n| inter_particle_averaged_pressure",
    "EDAC | ContinuityDensity | inter_particle_averaged_pressure\n+ AkinciFreeSurfaceCorrection",
    "EDAC | ContinuityDensity\n| density-matched pressure",
    "EDAC | ContinuityDensity | density-matched pressure\n+ AkinciFreeSurfaceCorrection",
    "EDAC | SummationDensity\n| inter_particle_averaged_pressure",
    "EDAC | SummationDensity | inter_particle_averaged_pressure\n+ AkinciFreeSurfaceCorrection",
    "EDAC | SummationDensity\n| density-matched pressure",
    "EDAC | SummationDensity | density-matched pressure\n+ AkinciFreeSurfaceCorrection",
    "IISPH | SummationDensity\n| implicit pressure",
    "IISPH | SummationDensity\n+ AkinciFreeSurfaceCorrection"]
model_order = ["CohesionForceAkinci", "SurfaceTensionAkinci"]
model_labels = ["CohesionForceAkinci\n(cohesion-only control)",
    "SurfaceTensionAkinci\n(cohesion + area minimization)"]

corrected_cases = count(case -> occursin("akinci_free_surface_correction", case),
                        case_order)
expected_rows = length(case_order) + corrected_cases
nrow(results) == expected_rows ||
    error("expected $expected_rows shootout rows in '$input_file', found $(nrow(results))")

best_configurations = Set{Tuple{String, String}}()
for method_results in groupby(results, [:sph_method, :density_calculator])
    best_row = method_results[argmin(method_results.late_time_mean_asphericity), :]
    push!(best_configurations, (String(best_row.case), String(best_row.model)))
end

function result_matrix(f; missing_value=NaN)
    return [begin
                rows = results[(results.case .== case) .& (results.model .== model), :]
                isempty(rows) ? missing_value : f(only(eachrow(rows)))
            end
            for case in case_order, model in model_order]
end

support_radius_cv_change = result_matrix() do row
    100 * (row.final_support_radius_cv / row.initial_support_radius_cv - 1)
end
radius_change = result_matrix(row -> 100 * (row.mean_radius_ratio - 1))
kinetic_energy = result_matrix(row -> row.kinetic_energy)
log_kinetic_energy = log10.(kinetic_energy)
is_best_configuration = result_matrix(; missing_value=false) do row
    return (String(row.case), String(row.model)) in best_configurations
end

function symmetric_range(values)
    limit = maximum(abs, filter(isfinite, vec(values)))
    return (-limit, limit)
end

function annotate_heatmap!(axis, values, formatter, text_color; checkmarks=nothing)
    for case_index in axes(values, 1), model_index in axes(values, 2)
        value = values[case_index, model_index]
        if !isfinite(value)
            text!(axis, model_index, case_index;
                  text="requires\ncorrection", align=(:center, :center), fontsize=11,
                  color=:gray35)
            continue
        end
        label = formatter(value)
        if !isnothing(checkmarks) && checkmarks[case_index, model_index]
            label = "\u2713 " * label
        end
        text!(axis, model_index, case_index;
              text=label, align=(:center, :center), fontsize=15,
              color=text_color(value))
    end

    hlines!(axis, [4.5, 12.5]; color=(:white, 0.9), linewidth=4)
    vlines!(axis, [1.5, 2.5]; color=(:white, 0.45), linewidth=1)
    return axis
end

quality_colormap = [:seagreen4, :honeydew, :mistyrose, :firebrick3]

figure = Figure(size=(2200, 1850), backgroundcolor=:white)
Label(figure[0, 1:3], "Akinci cube-to-sphere model shootout";
      fontsize=36, font=:bold, padding=(0, 0, 4, 0))
Label(figure[1, 1:3],
      "Only SurfaceTensionAkinci contains the surface-area-minimization term. " *
      "The cohesion-only column is a control; green shows metric direction, not solver pass/fail.";
      fontsize=18, color=:gray30, padding=(0, 0, 12, 0))

axis_options = (xticks=(1:length(model_order), model_labels),
                yticks=(1:length(case_order), case_labels),
                yreversed=true, xticklabelsize=14, xticklabelrotation=pi / 14,
                yticklabelsize=15,
                xgridvisible=false, ygridvisible=false)

axis_radial = Axis(figure[2, 1]; title="Shape response",
                   subtitle="DIRECTION: negative means the outer extent became more uniform",
                   axis_options...)
axis_radius = Axis(figure[2, 2]; title="Contraction direction",
                   subtitle="DIRECTION: negative means the mean radius contracted",
                   axis_options..., yticklabelsvisible=false)
axis_energy = Axis(figure[2, 3]; title="Dynamic response",
                   subtitle="DIAGNOSTIC: no universal good or bad value",
                   axis_options..., yticklabelsvisible=false)

support_radius_range = symmetric_range(support_radius_cv_change)
radius_range = symmetric_range(radius_change)
energy_range = extrema(filter(isfinite, vec(log_kinetic_energy)))

radial_plot = heatmap!(axis_radial, 1:length(model_order), 1:length(case_order),
                       permutedims(support_radius_cv_change);
                       colormap=quality_colormap, colorrange=support_radius_range)
radius_plot = heatmap!(axis_radius, 1:length(model_order), 1:length(case_order),
                       permutedims(radius_change);
                       colormap=quality_colormap, colorrange=radius_range)
energy_plot = heatmap!(axis_energy, 1:length(model_order), 1:length(case_order),
                       permutedims(log_kinetic_energy); colormap=:viridis,
                       colorrange=energy_range)

annotate_heatmap!(axis_radial, support_radius_cv_change,
                  value -> @sprintf("%+.1f", value),
                  value -> abs(value) > 0.45 * support_radius_range[2] ? :white : :black;
                  checkmarks=is_best_configuration)
annotate_heatmap!(axis_radius, radius_change, value -> @sprintf("%+.2f", value),
                  value -> abs(value) > 0.45 * radius_range[2] ? :white : :black)
annotate_heatmap!(axis_energy, kinetic_energy, value -> @sprintf("%.1e", value),
                  value -> log10(value) < sum(energy_range) / 2 ? :white : :black)

Colorbar(figure[3, 1], radial_plot; vertical=false,
         label="rounder  <--  support-radius CV change [%]  -->  less round",
         flipaxis=false)
Colorbar(figure[3, 2], radius_plot; vertical=false,
         label="contraction  <--  mean radius change [%]  -->  expansion",
         flipaxis=false)
Colorbar(figure[3, 3], energy_plot; vertical=false,
         label="log10(kinetic energy / J) | diagnostic only",
         flipaxis=false)

particle_count = only(unique(results.particle_count))
maximum_drift = maximum(results.center_of_mass_drift)
maximum_momentum = maximum(results.momentum_norm)
Label(figure[4, 1:3],
      "\u2713 marks the lowest mean asphericity over the final 20% of the run for each SPH " *
      "formulation + density calculator. A sphere has zero support-radius CV.";
      fontsize=16, color=:gray25, padding=(0, 0, 0, 6))
Label(figure[5, 1:3],
      "AkinciFreeSurfaceCorrection compensates for missing free-surface neighbors by " *
      "scaling surface-tension and viscosity forces; it does not modify pressure.";
      fontsize=16, color=:gray25, padding=(0, 0, 0, 8))
Label(figure[6, 1:3],
      "$expected_rows runs | $particle_count particles per run | " *
      "final time $(only(unique(results.final_time))) s | " *
      "max center drift $(@sprintf("%.2e", maximum_drift)) m | " *
      "max momentum $(@sprintf("%.2e", maximum_momentum)) kg m/s";
      fontsize=15, color=:gray40, padding=(0, 0, 0, 10))

colgap!(figure.layout, 22)
rowgap!(figure.layout, 18)
mkpath(dirname(output_file))
save(output_file, figure; px_per_unit=1.5)
println("Saved Akinci shootout graphic to $output_file")

figure

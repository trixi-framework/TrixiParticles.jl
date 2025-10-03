include("../validation_util.jl")

# Activate for interactive plot
# using GLMakie
using CairoMakie
using TrixiParticles
using TrixiParticles.CSV
using TrixiParticles.DataFrames
using TrixiParticles.JSON
using Glob
using Printf

# Set to save figures as SVG
save_figures = false
# Set to true to include simulation results in the `out` folder (if available)
include_sim_results = false

# Initial width of the fluid
H = 0.6
W = 2 * H

normalization_factor_time = sqrt(9.81 / H)
normalization_factor_pressure = 1000 * 9.81 * H

case_dir = joinpath(validation_dir(), "dam_break_2d")

edac_reference_files = joinpath.(case_dir,
                                 [
                                     "validation_reference_edac_40.json",
                                     "validation_reference_edac_80.json",
                                     "validation_reference_edac_400.json"
                                 ])
edac_sim_files = include_sim_results ?
                 glob("validation_result_dam_break_edac*.json", "out/") : []

merged_files = vcat(edac_reference_files, edac_sim_files)
edac_files = sort(merged_files, by=extract_number_from_filename)

wcsph_reference_files = joinpath.(case_dir,
                                  [
                                      "validation_reference_wcsph_40.json",
                                      "validation_reference_wcsph_80.json",
                                      "validation_reference_wcsph_400.json"
                                  ])
wcsph_sim_files = include_sim_results ?
                  glob("validation_result_dam_break_wcsph*.json", "out/") : []

merged_files = vcat(wcsph_reference_files, wcsph_sim_files)
wcsph_files = sort(merged_files, by=extract_number_from_filename)

# Read in external reference data
decourcy_P1 = CSV.read(joinpath(case_dir, "decourcy_pressure_sensor_P1.csv"), DataFrame)
decourcy_P2 = CSV.read(joinpath(case_dir, "decourcy_pressure_sensor_P2.csv"), DataFrame)
decourcy_P3 = CSV.read(joinpath(case_dir, "decourcy_pressure_sensor_P3.csv"), DataFrame)
decourcy_P4 = CSV.read(joinpath(case_dir, "decourcy_pressure_sensor_P4.csv"), DataFrame)

function plot_sensor_results(axs, files)
    for ax in axs
        ax.xlabel = "t(g / H)^0.5"
        ax.ylabel = "P/(ρgH)"
        xlims!(ax, 2, 7)
    end

    ylims!(axs[1], -0.2, 4.0)
    ylims!(axs[2], -0.5, 2.5)
    ylims!(axs[3], -0.5, 2)
    ylims!(axs[4], -0.5, 1.5)

    for (idx, json_file) in enumerate(files)
        println("Processing file: $json_file")

        json_data = JSON.parsefile(json_file)
        t = json_data["pressure_P1_fluid_1"]["time"] .* normalization_factor_time
        pressure_P1 = json_data["pressure_P1_fluid_1"]["values"] /
                      normalization_factor_pressure
        pressure_P2 = json_data["pressure_P2_fluid_1"]["values"] /
                      normalization_factor_pressure
        pressure_P3 = json_data["pressure_P3_fluid_1"]["values"] /
                      normalization_factor_pressure
        pressure_P4 = json_data["pressure_P4_fluid_1"]["values"] /
                      normalization_factor_pressure
        label_prefix = occursin("reference", json_file) ? "TrixiParticles " : ""
        res = extract_number_from_filename(json_file)

        lines!(axs[1], t, pressure_P1; label="$label_prefix H/$res",
               color=idx, colormap=:tab10, colorrange=(1, 10))
        lines!(axs[2], t, pressure_P2; label="$label_prefix H/$res",
               color=idx, colormap=:tab10, colorrange=(1, 10))
        lines!(axs[3], t, pressure_P3; label="$label_prefix H/$res",
               color=idx, colormap=:tab10, colorrange=(1, 10))
        lines!(axs[4], t, pressure_P4; label="$label_prefix H/$res",
               color=idx, colormap=:tab10, colorrange=(1, 10))
    end
end

n_sensors = 4
fig_sensors = Figure(size=(2400, 1000))
axs_edac = [Axis(fig_sensors[1, i], title="Sensor P$i (EDAC)")
            for i in 1:n_sensors]
axs_wcsph = [Axis(fig_sensors[3, i], title="Sensor P$i (WCSPH)")
             for i in 1:n_sensors]

function plot_reference(ax, time, data, label, color=:red, linestyle=:solid, linewidth=3)
    lines!(ax, time, data; color, linestyle, linewidth, label)
end

# Plot for pressure sensor P1
plot_reference(axs_edac[1], decourcy_P1.time, decourcy_P1.P1, "De Courcy 2024 (sim)")
plot_reference(axs_wcsph[1], decourcy_P1.time, decourcy_P1.P1, "De Courcy 2024 (sim)")

# Plot for pressure sensor P2
plot_reference(axs_edac[2], decourcy_P2.time, decourcy_P2.P2, "De Courcy 2024 (sim)")
plot_reference(axs_wcsph[2], decourcy_P2.time, decourcy_P2.P2, "De Courcy 2024 (sim)")

# Plot for pressure sensor P3
plot_reference(axs_edac[3], decourcy_P3.time, decourcy_P3.P3, "De Courcy 2024 (sim)")
plot_reference(axs_wcsph[3], decourcy_P3.time, decourcy_P3.P3, "De Courcy 2024 (sim)")

# Plot for pressure sensor P4
plot_reference(axs_edac[4], decourcy_P4.time, decourcy_P4.P4, "De Courcy 2024 (sim)")
plot_reference(axs_wcsph[4], decourcy_P4.time, decourcy_P4.P4, "De Courcy 2024 (sim)")

plot_sensor_results(axs_edac, edac_files)
plot_sensor_results(axs_wcsph, wcsph_files)

for (i, ax) in enumerate(axs_edac)
    Legend(fig_sensors[2, i], ax; tellwidth=false, orientation=:horizontal, valign=:top,
           nbanks=3)
end
for (i, ax) in enumerate(axs_wcsph)
    Legend(fig_sensors[4, i], ax; tellwidth=false, orientation=:horizontal, valign=:top,
           nbanks=3)
end

if save_figures
    save("dam_break_pressure.svg", fig_sensors)
else
    display(fig_sensors)
end

include("../validation_util.jl")

# Activate for interactive plot
# using GLMakie
using CairoMakie
using CSV
using DataFrames
using JSON
using Glob
using Printf
using TrixiParticles

# Initial width of the fluid
H = 0.6
W = 2 * H

normalization_factor_time = sqrt(9.81 / H)
normalization_factor_pressure = 1000 * 9.81 * H

edac_reference_files = glob("validation_reference_edac*.json",
                            "validation/dam_break_2d/")
edac_sim_files = glob("validation_result_dam_break_edac*.json",
                      "out/")

merged_files = vcat(edac_reference_files, edac_sim_files)
edac_files = sort(merged_files, by=extract_number_from_filename)

wcsph_reference_files = glob("validation_reference_wcsph*.json",
                             "validation/dam_break_2d/")
wcsph_sim_files = glob("validation_result_dam_break_wcsph*.json",
                       "out/")

merged_files = vcat(wcsph_reference_files, wcsph_sim_files)
wcsph_files = sort(merged_files, by=extract_number_from_filename)

surge_front = CSV.read("validation/dam_break_2d/exp_surge_front.csv", DataFrame)

exp_P1 = CSV.read("validation/dam_break_2d/exp_pressure_sensor_P1.csv", DataFrame)
exp_P2 = CSV.read("validation/dam_break_2d/exp_pressure_sensor_P2.csv", DataFrame)

sim_P1 = CSV.read("validation/dam_break_2d/sim_pressure_sensor_P1.csv", DataFrame)
sim_P2 = CSV.read("validation/dam_break_2d/sim_pressure_sensor_P2.csv", DataFrame)

n_sensors = 2
fig = Figure(size=(1200, 1200))
axs_edac = [Axis(fig[1, i], title="Sensor P$i with EDAC") for i in 1:n_sensors]
axs_wcsph = [Axis(fig[3, i], title="Sensor P$i with WCSPH") for i in 1:n_sensors]
ax_max_x_edac = Axis(fig[5, 1], title="Surge Front with EDAC")
ax_max_x_wcsph = Axis(fig[5, 2], title="Surge Front with WCSPH")

function plot_results(axs, ax_max, files)
    for ax in axs
        ax.xlabel = "Time"
        ax.ylabel = "Pressure"
        xlims!(ax, 0.0, 8.0)
        ylims!(ax, -0.1, 1.0)
    end

    # Define a regex to extract the sensor number from the key names
    sensor_number_regex = r"pressure_P(\d+)_fluid_\d+"

    for (file_number, json_file) in enumerate(files)
        json_data = JSON.parsefile(json_file)
        time = json_data["pressure_P1_fluid_1"]["time"] .* normalization_factor_time
        pressure_P1 = json_data["pressure_P1_fluid_1"]["values"] ./
                      normalization_factor_pressure
        pressure_P2 = json_data["pressure_P2_fluid_1"]["values"] ./
                      normalization_factor_pressure

        label_prefix = occursin("reference", json_file) ? "Reference" : ""

        lines!(axs[1], time, pressure_P1,
               label="$label_prefix dp=$(extract_resolution_from_filename(json_file))",
               color=file_number, colormap=:tab10, colorrange=(1, 10))
        lines!(axs[2], time, pressure_P2,
               label="$label_prefix dp=$(extract_resolution_from_filename(json_file))",
               color=file_number, colormap=:tab10, colorrange=(1, 10))
        value = json_data["max_x_coord_fluid_1"]
        lines!(ax_max, value["time"] .* sqrt(9.81), Float64.(value["values"]) ./ W,
               label="$label_prefix dp=$(extract_resolution_from_filename(json_file))")
    end
end

plot_results(axs_edac, ax_max_x_edac, edac_files)
plot_results(axs_wcsph, ax_max_x_wcsph, wcsph_files)

# Plot reference values
function plot_experiment(ax, time, data, label, color=:black, marker=:utriangle,
                         markersize=6)
    scatter!(ax, time, data; color, marker, markersize, label)
end

function plot_simulation(ax, time, data, label, color=:red, linestyle=:dash, linewidth=3)
    lines!(ax, time, data; color, linestyle, linewidth, label)
end

# Plot for Pressure Sensor P1
plot_experiment(axs_edac[1], exp_P1.time, exp_P1.P1, "Buchner 2002 (exp)")
plot_simulation(axs_edac[1], sim_P1.time, sim_P1.h320, "Marrone et al. 2011 (sim)")
plot_experiment(axs_wcsph[1], exp_P1.time, exp_P1.P1, "Buchner 2002 (exp)")
plot_simulation(axs_wcsph[1], sim_P1.time, sim_P1.h320, "Marrone et al. 2011 (sim)")

# Plot for Pressure Sensor P2
plot_experiment(axs_edac[2], exp_P2.time, exp_P2.P2, "Buchner 2002 (exp)")
plot_simulation(axs_edac[2], sim_P2.time, sim_P2.h320, "Marrone et al. 2011 (sim)")
plot_experiment(axs_wcsph[2], exp_P2.time, exp_P2.P2, "Buchner 2002 (exp)")
plot_simulation(axs_wcsph[2], sim_P2.time, sim_P2.h320, "Marrone et al. 2011 (sim)")

# Plot for Surge Front
for ax_max in [ax_max_x_edac, ax_max_x_wcsph]
    ax_max.xlabel = "Time"
    ax_max.ylabel = "Surge Front"
    xlims!(ax_max, 0.0, 3.0)
    ylims!(ax_max, 1, 3.0)
    plot_experiment(ax_max, surge_front.time, surge_front.surge_front,
                    "Martin and Moyce 1952 (exp)")
end

for (i, ax) in enumerate(axs_edac)
    Legend(fig[2, i], ax; tellwidth=false, orientation=:horizontal, valign=:top, nbanks=3)
end
for (i, ax) in enumerate(axs_wcsph)
    Legend(fig[4, i], ax; tellwidth=false, orientation=:horizontal, valign=:top, nbanks=3)
end
Legend(fig[6, 1], ax_max_x_edac, tellwidth=false, orientation=:horizontal, valign=:top,
       nbanks=2)
Legend(fig[6, 2], ax_max_x_wcsph, tellwidth=false, orientation=:horizontal, valign=:top,
       nbanks=2)

fig
# Uncomment to save the figure
# save("dam_break_validation.svg", fig)

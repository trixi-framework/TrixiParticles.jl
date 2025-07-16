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

# Set to save figures as SVG
save_figures = false
include_sim_results = false

# Initial width of the fluid
H = 0.6
W = 2 * H

normalization_factor_time = sqrt(9.81 / H)
normalization_factor_pressure = 1000 * 9.81 * H

case_dir = joinpath(validation_dir(), "dam_break_2d")

edac_reference_files = glob("validation_reference_edac*.json",
                            case_dir)
edac_sim_files = include_sim_results ?
                 glob("validation_result_dam_break_edac*.json",
                      "out/") : []

merged_files = vcat(edac_reference_files, edac_sim_files)
edac_files = sort(merged_files, by=extract_number_from_filename)

wcsph_reference_files = glob("validation_reference_wcsph*.json",
                             case_dir)
wcsph_sim_files = include_sim_results ?
                  glob("validation_result_dam_break_wcsph*.json",
                       "out/") : []

merged_files = vcat(wcsph_reference_files, wcsph_sim_files)
wcsph_files = sort(merged_files, by=extract_number_from_filename)

# Read in external reference data
surge_front = CSV.read(joinpath(case_dir, "exp_surge_front.csv"), DataFrame)

exp_P1 = CSV.read(joinpath(case_dir, "exp_pressure_sensor_P1.csv"), DataFrame)
exp_P2 = CSV.read(joinpath(case_dir, "exp_pressure_sensor_P2.csv"), DataFrame)

sim_P1 = CSV.read(joinpath(case_dir, "sim_pressure_sensor_P1.csv"), DataFrame)
sim_P2 = CSV.read(joinpath(case_dir, "sim_pressure_sensor_P2.csv"), DataFrame)

function plot_sensor_results(axs, files)
    for ax in axs
        ax.xlabel = "t(g / H)^0.5"
        ax.ylabel = "P/(ρ g H)"
        xlims!(ax, 2, 8)
        ylims!(ax, -0.2, 1.0)
    end

    for (idx, json_file) in enumerate(files)
        println("Processing file: $json_file")

        json_data = JSON.parsefile(json_file)
        t = json_data["interpolated_pressure_P1_fluid_1"]["time"] .*
            normalization_factor_time
        pressure_P1 = json_data["interpolated_pressure_P1_fluid_1"]["values"] /
                      normalization_factor_pressure
        pressure_P2 = json_data["interpolated_pressure_P2_fluid_1"]["values"] /
                      normalization_factor_pressure
        probe_P1 = json_data["particle_pressure_P1_boundary_1"]["values"] /
                   normalization_factor_pressure
        probe_P2 = json_data["particle_pressure_P2_boundary_1"]["values"] /
                   normalization_factor_pressure
        label_prefix = occursin("reference", json_file) ? "Ref. " : ""
        res = extract_resolution_from_filename(json_file)

        lines!(axs[1], t, pressure_P1; label="$label_prefix dp=$res",
               color=idx, colormap=:tab10, colorrange=(1, 10))
        lines!(axs[2], t, pressure_P2; label="$label_prefix dp=$res",
               color=idx, colormap=:tab10, colorrange=(1, 10))
        lines!(axs[3], t, probe_P1; label="$label_prefix dp=$res",
               color=idx, colormap=:tab10, colorrange=(1, 10))
        lines!(axs[4], t, probe_P2; label="$label_prefix dp=$res",
               color=idx, colormap=:tab10, colorrange=(1, 10))
    end
end

function plot_surge_results(ax, files)
    ax.xlabel = "Time [s]"
    ax.ylabel = "x / W"
    xlims!(ax, 0, 3)
    ylims!(ax, 1, 3)

    for (idx, json_file) in enumerate(files)
        json_data = JSON.parsefile(json_file)
        label_prefix = occursin("reference", json_file) ? "Ref. " : ""
        val = json_data["max_x_coord_fluid_1"]
        lines!(ax, val["time"] .* sqrt(9.81),
               Float64.(val["values"]) ./ W;
               label="$label_prefix dp=$(extract_resolution_from_filename(json_file))",
               color=idx, colormap=:tab10, colorrange=(1, 10))
    end

    # Experimental reference
    scatter!(ax, surge_front.time, surge_front.surge_front;
             color=:black, marker=:utriangle, markersize=6,
             label="Martin & Moyce 1952 (exp)")
end

# ------------------------------------------------------------
# 1) Pressure-sensor figure
# ------------------------------------------------------------
n_sensors = 4
fig_sensors = Figure(size=(2400, 1000))
axs_edac = [Axis(fig_sensors[1, i],
                 title=(i>2) ? "Boundary values at P$(i-2) (EDAC)" : "Sensor P$i (EDAC)")
            for i in 1:n_sensors]
axs_wcsph = [Axis(fig_sensors[3, i],
                  title=(i>2) ? "Boundary values at P$(i-2) (WCSPH)" : "Sensor P$i (WCSPH)")
             for i in 1:n_sensors]

plot_sensor_results(axs_edac, edac_files)
plot_sensor_results(axs_wcsph, wcsph_files)

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
plot_simulation(axs_edac[1], sim_P1.time, sim_P1.h320,
                "Marrone et al. 2011 dp=0.001875 (sim)")
plot_experiment(axs_wcsph[1], exp_P1.time, exp_P1.P1, "Buchner 2002 (exp)")
plot_simulation(axs_wcsph[1], sim_P1.time, sim_P1.h320,
                "Marrone et al. 2011 dp=0.001875 (sim)")
plot_experiment(axs_edac[3], exp_P1.time, exp_P1.P1, "Buchner 2002 (exp)")
plot_simulation(axs_edac[3], sim_P1.time, sim_P1.h320,
                "Marrone et al. 2011 dp=0.001875 (sim)")
plot_experiment(axs_wcsph[3], exp_P1.time, exp_P1.P1, "Buchner 2002 (exp)")
plot_simulation(axs_wcsph[3], sim_P1.time, sim_P1.h320,
                "Marrone et al. 2011 dp=0.001875 (sim)")

# Plot for Pressure Sensor P2
plot_experiment(axs_edac[2], exp_P2.time, exp_P2.P2, "Buchner 2002 (exp)")
plot_simulation(axs_edac[2], sim_P2.time, sim_P2.h320,
                "Marrone et al. 2011 dp=0.001875 (sim)")
plot_experiment(axs_wcsph[2], exp_P2.time, exp_P2.P2, "Buchner 2002 (exp)")
plot_simulation(axs_wcsph[2], sim_P2.time, sim_P2.h320,
                "Marrone et al. 2011 dp=0.001875 (sim)")
plot_experiment(axs_edac[4], exp_P2.time, exp_P2.P2, "Buchner 2002 (exp)")
plot_simulation(axs_edac[4], sim_P2.time, sim_P2.h320,
                "Marrone et al. 2011 dp=0.001875 (sim)")
plot_experiment(axs_wcsph[4], exp_P2.time, exp_P2.P2, "Buchner 2002 (exp)")
plot_simulation(axs_wcsph[4], sim_P2.time, sim_P2.h320,
                "Marrone et al. 2011 dp=0.001875 (sim)")

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

# ------------------------------------------------------------
# 2) Surge-front figure
# ------------------------------------------------------------
fig_surge = Figure(size=(1200, 400))
ax_surge_edac = Axis(fig_surge[1, 1], title="Surge Front – EDAC")
ax_surge_wcsph = Axis(fig_surge[1, 2], title="Surge Front – WCSPH")

plot_surge_results(ax_surge_edac, edac_files)
plot_surge_results(ax_surge_wcsph, wcsph_files)

Legend(fig_surge[2, 1], ax_surge_edac; orientation=:horizontal, valign=:top, nbanks=3)
Legend(fig_surge[2, 2], ax_surge_wcsph; orientation=:horizontal, valign=:top, nbanks=3)

if save_figures
    save("dam_break_surge_front.svg", fig_surge)
else
    display(fig_surge)
end

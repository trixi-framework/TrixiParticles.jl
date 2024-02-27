include("../validation_util.jl")

using Printf
using JSON
using CSV
using DataFrames
using Glob
# activate for interactive plot
#using GLMakie
using CairoMakie

# Initial width of the fluid
H = 0.6
W = 2 * H

normalization_factor_time = sqrt(9.81 / H)
normalization_factor_pressure = 1000 * 9.81 * H

edac_reference_files = glob("validation_reference_dam_break_edac*.json",
                            "validation/dam_break_2d/")
wcsph_reference_files = glob("validation_reference_dam_break_wcsph*.json",
                             "validation/dam_break_2d/")

# json_files = glob("dam_break_*.json", "validation/dam_break_2d/")

surge_front = CSV.read("validation/dam_break_2d/exp_surge_front.csv", DataFrame)

exp_P1 = CSV.read("validation/dam_break_2d/exp_pressure_sensor_P1.csv", DataFrame)
exp_P2 = CSV.read("validation/dam_break_2d/exp_pressure_sensor_P2.csv", DataFrame)

sim_P1 = CSV.read("validation/dam_break_2d/sim_pressure_sensor_P1.csv", DataFrame)
sim_P2 = CSV.read("validation/dam_break_2d/sim_pressure_sensor_P2.csv", DataFrame)

n_sensors = 2
fig = Figure(size=(1200, 1200))
axs_edac = [Axis(fig[1, i], title="Sensor P$i with EDAC") for i in 1:n_sensors]
axs_wcsph = [Axis(fig[3, i], title="Sensor P$i with WCSPH") for i in 1:n_sensors]
ax_max_x = Axis(fig[5, 1], title="Surge Front")

function plot_results(axs, files)
    for ax in axs
        ax.xlabel = "Time"
        ax.ylabel = "Pressure"
        xlims!(ax, 0.0, 8.0)
        ylims!(ax, -0.1, 1.0)
    end

    # Define a regex to extract the sensor number from the key names
    sensor_number_regex = r"pressure_P(\d+)_fluid_\d+"

    file_number = 1
    for json_file in files
        json_data = JSON.parsefile(json_file)
        for (key, value) in json_data
            if occursin(sensor_number_regex, key)
                matches = match(sensor_number_regex, key)
                sensor_index = parse(Int, matches[1])
                if sensor_index >= 1 && sensor_index <= length(axs)
                    time = value["time"] .* normalization_factor_time
                    values = value["values"] ./ normalization_factor_pressure
                    lines!(axs[sensor_index], time, values, color=file_number,
                           label="dp=" *
                                 convert_to_float(split(replace(basename(json_file),
                                                                ".json" => ""), "_")[end]),
                           colormap=:tab10, colorrange=(1, 10))
                else
                    println("Sensor index $sensor_index was not plotted.")
                end
            end
        end
        if haskey(json_data, "max_x_coord_fluid_1")
            value = json_data["max_x_coord_fluid_1"]
            time = value["time"] .* sqrt(9.81)
            values = Float64.(value["values"]) ./ W
            lines!(ax_max_x, time, values,
                   label="dp=" * convert_to_float(split(replace(basename(json_file),
                                                        ".json" => ""), "_")[end]))
        end
        file_number += 1
    end
end

plot_results(axs_edac, edac_reference_files)
plot_results(axs_wcsph, wcsph_reference_files)

ax_max_x.xlabel = "Time"
ax_max_x.ylabel = "Surge Front"
xlims!(ax_max_x, 0.0, 3.0)
ylims!(ax_max_x, 1, 3.0)

# Plot reference values
scatter!(axs_edac[1], exp_P1.time, exp_P1.P1, color=:black, marker=:utriangle, markersize=6,
         label="Buchner 2002 (exp)")
lines!(axs_edac[1], sim_P1.time, sim_P1.h320, color=:red, linestyle=:dash, linewidth=3,
       label="Marrone et al. 2011 (sim)")
scatter!(axs_wcsph[1], exp_P1.time, exp_P1.P1, color=:black, marker=:utriangle,
         markersize=6,
         label="Buchner 2002 (exp)")
lines!(axs_wcsph[1], sim_P1.time, sim_P1.h320, color=:red, linestyle=:dash, linewidth=3,
       label="Marrone et al. 2011 (sim)")

scatter!(axs_edac[2], exp_P2.time, exp_P2.P2, color=:black, marker=:utriangle, markersize=6,
         label="Buchner 2002 (exp)")
lines!(axs_edac[2], sim_P2.time, sim_P2.h320, color=:red, linestyle=:dash, linewidth=3,
       label="Marrone et al. 2011 (sim)")
scatter!(axs_wcsph[2], exp_P2.time, exp_P2.P2, color=:black, marker=:utriangle,
         markersize=6,
         label="Buchner 2002 (exp)")
lines!(axs_wcsph[2], sim_P2.time, sim_P2.h320, color=:red, linestyle=:dash, linewidth=3,
       label="Marrone et al. 2011 (sim)")

scatter!(ax_max_x, surge_front.time, surge_front.surge_front, color=:black,
         marker=:utriangle, markersize=6,
         label="Martin and Moyce 1952 (exp)")

for (i, ax) in enumerate(axs_edac)
    Legend(fig[2, i], ax; tellwidth=false, orientation=:horizontal, valign=:top, nbanks=3)
end
for (i, ax) in enumerate(axs_wcsph)
    Legend(fig[4, i], ax; tellwidth=false, orientation=:horizontal, valign=:top, nbanks=3)
end
Legend(fig[6, 1], ax_max_x, tellwidth=false, orientation=:horizontal, valign=:top, nbanks=2)

fig
# uncomment to save the figure
#save("dam_break_validation.svg", fig)

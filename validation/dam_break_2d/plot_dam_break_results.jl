include("../validation_util.jl")

using Printf
using JSON
using Glob
# activate for interactive plot
#using GLMakie
using CairoMakie

normalization_factor_time = sqrt(9.81 / 0.6)
normalization_factor_pressure = 1000 * 9.81

sensor_data = Dict("P1" => [], "P2" => [], "P3" => [])

json_files = glob("dam_break_*.json", "validation/dam_break_2d/")

fig = Figure(size=(1200, 800))
axs = [Axis(fig[1, i], title="Sensor P$i") for i in 1:3]
ax_max_x = Axis(fig[3, 1], title = "Fluid Progress")

# Set common axis labels and limits
for ax in axs
    ax.xlabel = "Time"
    ax.ylabel = "Pressure"
    xlims!(ax, 0.0, 8.0)
    ylims!(ax, -0.1, 1.0)
end

ax_max_x.xlabel = "Time"
ax_max_x.ylabel = "Fluid Progress"
xlims!(ax_max_x, 0.0, 3.0)
ylims!(ax_max_x, 2, 6)

# Define a regex to extract the sensor number from the key names
sensor_number_regex = r"pressure_P(\d+)_fluid_\d+"

file_number = 1
for json_file in json_files
    json_data = JSON.parsefile(json_file)
    for (key, value) in json_data
        if occursin(sensor_number_regex, key)
            matches = match(sensor_number_regex, key)
            sensor_index = parse(Int, matches[1])
            if sensor_index >= 1 && sensor_index <= length(axs)
                time = value["time"] .* normalization_factor_time
                values = value["values"] ./ normalization_factor_pressure
                lines!(axs[sensor_index], time, values, color=file_number,
                       label="dp="*convert_to_float(split(replace(basename(json_file),
                                                            ".json" => ""), "_")[end]),
                       colormap=:tab10, colorrange=(1, 10))
            else
                println("Sensor index $sensor_index out of expected range for key $key")
            end
        end
    end
    if haskey(json_data, "max_x_coord_fluid_1")
        value = json_data["max_x_coord_fluid_1"]
        time = value["time"] .* normalization_factor_time
        values = Float64.(value["values"])
        lines!(ax_max_x, time, values,
        label="dp="*convert_to_float(split(replace(basename(json_file),
                                             ".json" => ""), "_")[end]))
    end
    global file_number += 1
end

for (i, ax) in enumerate(axs)
    Legend(fig[2, i], ax; tellwidth=false, orientation=:horizontal, valign=:top)
end
Legend(fig[4, 1], ax_max_x, tellwidth = false, orientation=:horizontal, valign=:top)

fig

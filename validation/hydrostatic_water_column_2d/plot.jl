include("../validation_util.jl")
using CairoMakie
using JSON
using Glob
using Statistics
using Printf
using TrixiParticles

# Helper function: find the key that starts with the given base name.
function find_key(json_data::Dict{String,Any}, base::String)
    ks = filter(k -> startswith(k, base), keys(json_data))
    isempty(ks) && error("No key starting with \"$base\" found in JSON data")
    return first(ks)
end

# --- Gather simulation result files for EDAC and WCSPH methods
edac_files = sort(glob("validation_result_hydrostatic_water_column_2d_edac*.json", "out/"),
                  by=extract_number_from_filename)
wcsph_files = sort(glob("validation_result_hydrostatic_water_column_2d_wcsph*.json", "out/"),
                   by=extract_number_from_filename)

# Define explicit color ranges.
edac_range = (1, max(length(edac_files), 2))
wcsph_range = (1, max(length(wcsph_files), 2))

# --- Compute global y-axis limits (from both simulation and analytical data)
all_y = Float64[]
for file in vcat(edac_files, wcsph_files)
    json_data = JSON.parsefile(file)
    defl_key = find_key(json_data, "y_deflection")
    analytical_key = find_key(json_data, "analytical_sol")
    append!(all_y, json_data[defl_key]["values"])
    append!(all_y, json_data[analytical_key]["values"])
end
global_ymin = minimum(all_y)
global_ymax = maximum(all_y)

# --- Create the figure (using size instead of resolution)
fig = Figure(size = (1200, 600))
ax_edac   = Axis(fig[1, 1], title="Hydrostatic Water Column: EDAC")
ax_wcsph  = Axis(fig[1, 2], title="Hydrostatic Water Column: WCSPH")

# --- Function to plot one dataset (simulation and analytical curves plus simulation average)
function plot_dataset!(ax, json_file, col_range, file_number)
    json_data = JSON.parsefile(json_file)
    defl_key = find_key(json_data, "y_deflection")
    analytical_key = find_key(json_data, "analytical_sol")
    time_vals  = json_data[defl_key]["time"]
    sim_vals   = json_data[defl_key]["values"]
    anal_vals  = json_data[analytical_key]["values"]

    # Compute indices for times > 0.5.
    inds = findall(t -> t > 0.5, time_vals)
    avg_sim = mean(sim_vals[inds])

    # Plot simulation curve (using the colormap)
    lines!(ax, time_vals, sim_vals;
           label = "Simulation",
           color = :blue, linewidth = 2)
    # Plot analytical curve in black (it is already constant)
    lines!(ax, time_vals, anal_vals;
           label = "Analytical",
           color = :black, linestyle = :dash, linewidth = 4)

    # Plot horizontal line for simulation time-average (t > 0.5)
    lines!(ax, [time_vals[1], last(time_vals)], [avg_sim, avg_sim];
           color = :red, linestyle = :dot, linewidth = 4,
           label = "Sim avg (t>0.5)")
end

# --- Plot EDAC results
for (file_number, json_file) in enumerate(edac_files)
    plot_dataset!(ax_edac, json_file, edac_range, file_number)
end

# --- Plot WCSPH results
for (file_number, json_file) in enumerate(wcsph_files)
    plot_dataset!(ax_wcsph, json_file, wcsph_range, file_number)
end

# --- Set same y-axis limits for both axes
ylims!(ax_edac, global_ymin, global_ymax)
ylims!(ax_wcsph, global_ymin, global_ymax)

# --- Set axis labels and add legends
for ax in (ax_edac, ax_wcsph)
    ax.xlabel = "Time"
    ax.ylabel = "Vertical Deflection"
    axislegend(ax; tellwidth=false, orientation=:horizontal)
end

fig
# Uncomment to save the figure:
save("hydrostatic_water_column_validation.svg", fig)

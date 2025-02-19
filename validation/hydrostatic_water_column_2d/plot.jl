include("../validation_util.jl")
using CairoMakie
using JSON
using Glob
using Statistics
using Printf
using TrixiParticles

case_dir = joinpath(validation_dir(), "dam_break_2d")

edac_files = sort(glob("validation_reference_edac*.json", case_dir),
                  by=extract_number_from_filename)
wcsph_files = sort(glob("validation_reference_wcsph*.json", case_dir),
                   by=extract_number_from_filename)


edac_sim_files = sort(glob("validation_result_hyd_edac*.json", "out/"),
                   by=extract_number_from_filename)
wcsph_sim_files = sort(glob("validation_result_hyd_wcsph*.json", "out/"),
                    by=extract_number_from_filename)

# Define explicit color ranges.
edac_range = (1, max(length(edac_files), 2))
wcsph_range = (1, max(length(wcsph_files), 2))

# --- Compute global y-axis limits (from both simulation and analytical data)
all_y = Float64[]
for file in vcat(edac_files, wcsph_files)
    json_data = JSON.parsefile(file)
    println("Open json file ", file)
    append!(all_y, json_data["y_deflection_solid_1"]["values"])
    append!(all_y, json_data["analytical_solution"]["values"])
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
    time_vals  = json_data["y_deflection_solid_1"]["time"]
    sim_vals   = json_data["y_deflection_solid_1"]["values"]
    anal_vals  = json_data["analytical_solution"]["values"]

    # Compute indices for times > 0.5.
    inds = findall(t -> t > 0.5, time_vals)
    avg_sim = mean(sim_vals[inds])

    error = avg_sim - anal_vals[1]
    println("Abs. Error: ", error)
    println("Rel. Error: ", error/anal_vals[1])

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

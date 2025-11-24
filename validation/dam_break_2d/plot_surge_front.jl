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

case_dir = joinpath(validation_dir(), "dam_break_2d")

edac_reference_files = joinpath.(case_dir,
                                 [
                                     "validation_reference_edac_40.json",
                                     "validation_reference_edac_80.json"
                                 ])
edac_sim_files = include_sim_results ?
                 glob("validation_result_dam_break_edac*.json", "out/") : []

merged_files = vcat(edac_reference_files, edac_sim_files)
edac_files = sort(merged_files, by=extract_number_from_filename)

wcsph_reference_files = joinpath.(case_dir,
                                  [
                                      "validation_reference_wcsph_40.json",
                                      "validation_reference_wcsph_80.json"
                                  ])
wcsph_sim_files = include_sim_results ?
                  glob("validation_result_dam_break_wcsph*.json", "out/") : []

merged_files = vcat(wcsph_reference_files, wcsph_sim_files)
wcsph_files = sort(merged_files, by=extract_number_from_filename)

# Read in external reference data
surge_front = CSV.read(joinpath(case_dir, "exp_surge_front.csv"), DataFrame)

function plot_surge_results(ax, files)
    ax.xlabel = "Time [s]"
    ax.ylabel = "x / W"
    xlims!(ax, 0, 3)
    ylims!(ax, 1, 3)

    for (idx, json_file) in enumerate(files)
        json_data = JSON.parsefile(json_file)
        label_prefix = occursin("reference", json_file) ? "TrixiParticles " : ""
        val = json_data["max_x_coord_fluid_1"]
        lines!(ax, val["time"] .* sqrt(9.81),
               Float64.(val["values"]) ./ W;
               label="$label_prefix H/$(extract_number_from_filename(json_file))",
               color=idx, colormap=:tab10, colorrange=(1, 10))
    end

    # Experimental reference
    scatter!(ax, surge_front.time, surge_front.surge_front;
             color=:black, marker=:utriangle, markersize=6,
             label="Martin & Moyce 1952 (exp)")
end

fig_surge = Figure(size=(1200, 400))
ax_surge_edac = Axis(fig_surge[1, 1], title="Surge Front – EDAC")
ax_surge_wcsph = Axis(fig_surge[1, 2], title="Surge Front – WCSPH")

plot_surge_results(ax_surge_edac, edac_files)
plot_surge_results(ax_surge_wcsph, wcsph_files)

Legend(fig_surge[2, 1], ax_surge_edac; orientation=:horizontal, valign=:top, nbanks=3)
Legend(fig_surge[2, 2], ax_surge_wcsph; orientation=:horizontal, valign=:top, nbanks=3)

if save_figures
    save("dam_break_surge.svg", fig_surge)
else
    display(fig_surge)
end

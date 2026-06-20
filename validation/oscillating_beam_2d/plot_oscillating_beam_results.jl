include("../validation_util.jl")

# Activate for interactive plots
# using GLMakie
using CairoMakie
using CSV
using DataFrames
using JSON
using Glob
using TrixiParticles

const RESOLUTIONS = (5, 9, 17, 33, 65)
const COLORS = ("#ff0000", "#f5a000", "#800080", "#3caf75")
const TIME_LIMITS = (0.35, 0.55)
const DEFLECTION_LIMITS = (-0.235, -0.105)

function load_tip_deflection(file_name)
    json_data = JSON.parsefile(file_name)
    data = json_data["deflection_y_structure_1"]

    return Float64.(data["time"]), Float64.(data["values"])
end

function file_for_resolution(files, resolution)
    index = findfirst(file -> extract_number_from_filename(file) == resolution, files)
    return isnothing(index) ? nothing : files[index]
end

reference_directory = joinpath(validation_dir(), "oscillating_beam_2d")
reference_files = glob("validation_reference_*.json", reference_directory)
simulation_files = glob("validation_run_oscillating_beam_2d_*.json", "out")
turek = CSV.read(joinpath(reference_directory, "reference_turek.csv"), DataFrame)

set_theme!(Theme(fontsize=28, fonts=(; regular="TeX Gyre Termes")))

fig = Figure(size=(1080, 760), figure_padding=(20, 25, 15, 15))
ax = Axis(fig[1, 1];
          xlabel="Time (s)",
          ylabel="Tip Y - Deflection (m)",
          limits=(TIME_LIMITS, DEFLECTION_LIMITS),
          xticks=0.35:0.05:0.55,
          yticks=-0.13:0.01:-0.11,
          xlabelsize=34,
          ylabelsize=34,
          xticklabelsize=30,
          yticklabelsize=30,
          spinewidth=2.5,
          xtickwidth=2.5,
          ytickwidth=2.5,
          xticksize=10,
          yticksize=10)

line_plots = []
marker_plots = []

for (resolution, color) in zip(RESOLUTIONS, COLORS)
    reference_file = file_for_resolution(reference_files, resolution)
    isnothing(reference_file) &&
        error("Reference data for resolution $resolution not found")

    reference_time, reference_deflection = load_tip_deflection(reference_file)
    marker_plot = scatter!(ax, reference_time, reference_deflection;
                           color=:transparent,
                           strokecolor=color,
                           strokewidth=3,
                           marker=:circle,
                           markersize=15)
    push!(marker_plots, marker_plot)

    simulation_file = file_for_resolution(simulation_files, resolution)
    if isnothing(simulation_file)
        @warn "Simulation data for resolution $resolution not found; plotting the reference data as the solid curve"
        simulation_time = reference_time
        simulation_deflection = reference_deflection
    else
        simulation_time, simulation_deflection = load_tip_deflection(simulation_file)
    end

    line_plot = lines!(ax, simulation_time, simulation_deflection;
                       color, linewidth=4)
    push!(line_plots, line_plot)
end

turek_plot = lines!(ax, turek.time, turek.Uy;
                    color=:black, linestyle=:dash, linewidth=3)

# Four empty entries make the two reference labels occupy the first two rows
# of the right legend column.
empty_entry = LineElement(color=:transparent)
legend_elements = [line_plots...,
                   MarkerElement(marker=:circle, color=:transparent,
                                 strokecolor=:black, strokewidth=3,
                                 markersize=15),
                   turek_plot,
                   empty_entry,
                   empty_entry]
legend_labels = [L"t_s/dp = 8",
                 L"t_s/dp = 16",
                 L"t_s/dp = 32",
                 L"t_s/dp = 64",
                 "O'Connor and Rogers, 2021",
                 "Turek & Hron, 2007",
                 "",
                 ""]

Legend(fig[1, 1], legend_elements, legend_labels;
       nbanks=2,
       tellwidth=false,
       tellheight=false,
       halign=:center,
       valign=:top,
       margin=(20, 20, 8, 8),
       padding=(8, 8, 5, 5),
       patchsize=(35, 20),
       rowgap=1,
       colgap=16,
       framevisible=true,
       framewidth=1,
       backgroundcolor=:white)

# save("validation_oscillating_beam_2d.png", fig, px_per_unit=2)
fig

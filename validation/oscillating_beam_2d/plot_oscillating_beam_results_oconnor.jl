# Plot the tip y-deflection of the TrixiParticles.jl reference results against
# the results by O'Connor and Rogers (2021) and Turek and Hron (2006).

include("../validation_util.jl")

using Plots
using LaTeXStrings
using CSV
using DataFrames
using JSON
using TrixiParticles

elastic_plate = (length=0.35, thickness=0.02)

# Number of particles over the plate thickness. Note that `t_s / dp = resolution - 1`.
resolutions = (5, 9, 17, 33, 65)
colors = palette(:Set1_5)[[1, 5, 4, 3, 2]]
time_limits = (0.35, 0.55)
deflection_limits = (-0.135, -0.105)

function restrict_to_time_limits(times, values)
    mask = (first(time_limits) .<= times) .& (times .<= last(time_limits))
    return times[mask], values[mask]
end

# Load the Turek and Hron reference data
ref_turek = CSV.read(joinpath(validation_dir(), "oscillating_beam_2d",
                              "reference_turek.csv"), DataFrame)

# Load the O'Connor and Rogers reference data. Column `resN` contains the results for
# `t_s / dp = N`.
ref_oconnor = CSV.read(joinpath(validation_dir(), "oscillating_beam_2d",
                                "reference_oconnor.csv"), DataFrame)

function load_tip_deflection_y(file_name)
    json_data = JSON.parsefile(file_name)
    key = only(filter(key -> occursin(r"deflection_y_structure_\d+", key),
                      keys(json_data)))
    data = json_data[key]

    return Float64.(data["time"]), Float64.(data["values"])
end

p = plot(; xlabel="Time [s]", ylabel="Tip Y-Deflection [m]",
         xlims=time_limits, ylims=deflection_limits,
         xticks=0.35:0.05:0.55, yticks=-0.13:0.005:-0.11,
         framestyle=:box, legend=:top, legend_columns=2, dpi=400,
         left_margin=8Plots.mm, bottom_margin=5Plots.mm, right_margin=5Plots.mm)

# Legend entries of the reference data. The legend is filled row-wise with two columns,
# so these are interleaved with the resolutions to end up in the right column.
function plot_oconnor_legend!(p)
    # Don't use NaN, as this messes up the legend with PGFPlotsX.
    scatter!(p, [0], [0]; label="O'Connor and Rogers (2021)",
             markershape=:circle, markersize=4, markercolor=:white,
             markerstrokecolor=:black, markerstrokewidth=2.5)
end

function plot_turek!(p)
    times, deflection = restrict_to_time_limits(ref_turek.time, ref_turek.Uy)
    plot!(p, times, deflection; color=:black, linestyle=:dot, linewidth=2.5,
          label="Turek & Hron (2006)")
end

function plot_empty_legend!(p)
    # Don't use NaN, as this messes up the legend with PGFPlotsX.
    plot!(p, [0], [0]; label=" ", linealpha=0)
end

right_column = (plot_oconnor_legend!, plot_turek!,
                plot_empty_legend!, plot_empty_legend!, plot_empty_legend!)

for (resolution, color, plot_right_column!) in zip(resolutions, colors, right_column)
    file_name = joinpath(validation_dir(), "oscillating_beam_2d",
                         "validation_reference_$resolution.json")
    times, deflection = load_tip_deflection_y(file_name)
    times, deflection = restrict_to_time_limits(times, deflection)

    ratio = resolution - 1
    plot!(p, times, deflection; color, linewidth=2.5, label=latexstring("t_s/dp = $ratio"))
    plot_right_column!(p)

    scatter!(p, ref_oconnor.X, ref_oconnor[!, "res$ratio"]; label="",
             markershape=:circle, markersize=4, markercolor=:white,
             markerstrokecolor=color, markerstrokewidth=2.5)
end

p

# Plot the maximum tip y-deflection during the first swing of the beam over the resolution
# for both validation setups and compare it with the reference solution
# by Turek & Hron (2006).

include("../validation_util.jl")

using Plots
using CSV
using DataFrames
using JSON
using Glob
using TrixiParticles

# The first minimum of the tip y-deflection is at t ≈ 0.455.
first_swing_end = 0.7

function first_swing_minimum(times, values)
    return minimum(values[times .< first_swing_end])
end

# Load the Turek & Hron reference data.
ref_turek = CSV.read(joinpath(validation_dir(), "oscillating_beam_2d",
                              "reference_turek.csv"), DataFrame)
ref_minimum = first_swing_minimum(ref_turek.time, ref_turek.Uy)

function load_first_swing_minimum(file_name)
    json_data = JSON.parsefile(file_name)
    key = only(filter(key -> occursin(r"deflection_y_structure_\d+", key),
                      keys(json_data)))
    data = json_data[key]

    return first_swing_minimum(Float64.(data["time"]), Float64.(data["values"]))
end

# Use the number of particles across the beam thickness for both setups.
# Note that `thickness / particle_spacing = n_particles_beam_y - 1` for the O'Connor setup
# (particles on the shell) and `thickness / particle_spacing = n_particles_beam_y` for the
# Turek setup (cell-centered particles).
colors = palette(:Set1_5)
setups = ((name="oconnor", label="O'Connor setup (on-shell, beam extends into cylinder)",
           color=colors[1], markershape=:circle),
          (name="turek", label="Turek setup (cell-centered, beam glued to cylinder)",
           color=colors[2], markershape=:rect))

n_particles = 2 .^ (2:9) .+ 1
p = plot(; xlabel="Particles Across Beam Thickness", ylabel="Max. Tip Y-Deflection [m]",
         xscale=:log2, xticks=(n_particles, string.(n_particles)), xlims=(4.3, 600),
         framestyle=:box, legend=:bottomright, dpi=400,
         left_margin=8Plots.mm, bottom_margin=5Plots.mm, right_margin=5Plots.mm)

hline!(p, [ref_minimum]; color=:black, linestyle=:dot, linewidth=2.5,
       label="Turek & Hron (2006)")

# Add the O'Connor legend entry here to make the label appear second,
# but plot the markers after the setups below to draw them on top.
oconnor_style = (markershape=:x, markerstrokewidth=2, markersize=5, markercolor=:black)
scatter!(p, [NaN], [NaN]; oconnor_style..., label="O'Connor & Rogers (2021)")

for setup in setups
    files = glob("validation_reference_$(setup.name)_*.json",
                 joinpath(validation_dir(), "oscillating_beam_2d"))
    isempty(files) && continue

    sort!(files, by=extract_number_from_filename)
    n_particles_beam_y = extract_number_from_filename.(files)
    minima = load_first_swing_minimum.(files)

    plot!(p, n_particles_beam_y, minima; color=setup.color, linewidth=2.5,
          markershape=setup.markershape, markersize=4, markercolor=:white,
          markerstrokecolor=setup.color, markerstrokewidth=2.5, label=setup.label)
end

# Load the O'Connor & Rogers (2021) reference data around the first minimum.
# The columns `resN` correspond to `thickness / particle_spacing = N`, so with particles
# on the shell, there are `N + 1` particles across the beam thickness.
ref_oconnor = CSV.read(joinpath(validation_dir(), "oscillating_beam_2d",
                                "reference_oconnor.csv"), DataFrame)
res_columns = filter(name -> startswith(name, "res"), names(ref_oconnor))
n_particles_oconnor = [parse(Int, chopprefix(name, "res")) + 1 for name in res_columns]
minima_oconnor = [first_swing_minimum(ref_oconnor.X, ref_oconnor[!, name])
                  for name in res_columns]
scatter!(p, n_particles_oconnor, minima_oconnor; oconnor_style..., label="")

p

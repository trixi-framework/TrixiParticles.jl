using TrixiParticles
using CSV, DataFrames, Plots

output_directory = "out"

# ======================================================================================
# ==== Read results
data = CSV.read(joinpath(output_directory, "values.csv"), DataFrame)

times = data[!, "time"]
times_ref = [1.6, 3.2, 4.8, 6.4, 8.0] .* 1e-3
data_indices = findall(t -> t in times_ref, round.(times, digits=4))
radial_displacements = [eval(Meta.parse(str))
                        for str in data[!, "radial_displacement_structure_1"]][data_indices]
pressures_along_axis = [eval(Meta.parse(str))
                        for str in data[!, "pressure_along_axis_fluid_1"]][data_indices]

plot_range = range(0, 0.1, length=length(first(radial_displacements)))
displacements = view(stack(radial_displacements), :, :)
pressures = view(stack(pressures_along_axis), :, :)
label_ = ["1.6" "3.2" "4.8" "6.4" "8"] .* " ms"
line_colors = cgrad(:coolwarm, length(times_ref), categorical=true)

p1 = plot(plot_range, displacements, label=label_, linewidth=3,
          ylims=(-1e-4, 5e-4), xlims=(0, 0.1),
          palette=line_colors.colors, legend_position=:outerright, size=(750, 400))
yaxis!(p1, ylabel="Radial displacement (m)")
xaxis!(p1, xlabel="Distance (m)")
plot!(left_margin=5Plots.mm)
plot!(bottom_margin=5Plots.mm)

p2 = plot(plot_range, pressures, label=label_, linewidth=3,
          ylims=(-1500, 7500), xlims=(0, 0.1),
          palette=line_colors.colors, legend_position=:outerright, size=(750, 400))
yaxis!(p2, ylabel="Pressure along the centerline")
xaxis!(p2, xlabel="Distance (m)")
plot!(left_margin=5Plots.mm)
plot!(bottom_margin=5Plots.mm)

plot(p1, p2, layout=(1, 2), size=(900, 350))

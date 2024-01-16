# Example for using interpolation
#######################################################################################
using TrixiParticles
# this needs to be commented out to use PythonPlot
using Plots
using Plots.PlotMeasures

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "rectangular_tank_2d.jl"))

# Interpolation parameters
interpolation_start = [0.0, 0.0]
interpolation_end = [1.0, 1.0]
resolution = 0.005

# Original plane
original_plane = interpolate_plane(interpolation_start, interpolation_end, resolution, semi,
                                   fluid_system, sol)
original_x = [point[1] for point in original_plane.coord]
original_y = [point[2] for point in original_plane.coord]
original_density = original_plane.density

# Plane with double smoothing length
double_smoothing_plane = interpolate_plane(interpolation_start, interpolation_end,
                                           resolution, semi, fluid_system, sol,
                                           smoothing_length=2.0 * smoothing_length)
double_x = [point[1] for point in double_smoothing_plane.coord]
double_y = [point[2] for point in double_smoothing_plane.coord]
double_density = double_smoothing_plane.density

# Plane with half smoothing length
half_smoothing_plane = interpolate_plane(interpolation_start, interpolation_end, resolution,
                                         semi, fluid_system, sol,
                                         smoothing_length=0.5 * smoothing_length)
half_x = [point[1] for point in half_smoothing_plane.coord]
half_y = [point[2] for point in half_smoothing_plane.coord]
half_density = half_smoothing_plane.density

# Instead of using Plots.jl one can also use PythonPlot which uses matplotlib
# using PythonPlot

# # Initialize figure with three subplots
# fig, (subplot1, subplot2, subplot3) = subplots(1, 3, figsize=(15, 5))

# # Plot for original plane
# scatter1 = subplot1.scatter(original_x, original_y, c=original_density, cmap="viridis",
#                             marker="o", vmin=1000, vmax=1010)
# subplot1.set_xlabel("X Coordinate")
# subplot1.set_ylabel("Y Coordinate")
# subplot1.set_title("Density Distribution")
# fig.colorbar(scatter1, ax=subplot1, label="Density")

# # Plot for plane with double smoothing length
# scatter2 = subplot2.scatter(double_x, double_y, c=double_density, cmap="viridis",
#                             marker="o", vmin=1000, vmax=1010)
# subplot2.set_xlabel("X Coordinate")
# subplot2.set_ylabel("Y Coordinate")
# subplot2.set_title("Density with 2x Smoothing Length")
# fig.colorbar(scatter2, ax=subplot2, label="Density")

# # Plot for plane with half smoothing length
# scatter3 = subplot3.scatter(half_x, half_y, c=half_density, cmap="viridis", marker="o",
#                             vmin=1000, vmax=1010)
# subplot3.set_xlabel("X Coordinate")
# subplot3.set_ylabel("Y Coordinate")
# subplot3.set_title("Density with 0.5x Smoothing Length")
# fig.colorbar(scatter3, ax=subplot3, label="Density")

# fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
# plotshow()

scatter1 = scatter(original_x, original_y, zcolor=original_density, marker=:circle,
                   markersize=2, markercolor=:viridis, markerstrokewidth=0,
                   clim=(1000, 1010), colorbar=true, legend=false)
scatter2 = scatter(double_x, double_y, zcolor=double_density, marker=:circle, markersize=2,
                   markercolor=:viridis, markerstrokewidth=0, clim=(1000, 1010),
                   colorbar=true, legend=false)
scatter3 = scatter(half_x, half_y, zcolor=half_density, marker=:circle, markersize=2,
                   markercolor=:viridis, markerstrokewidth=0, clim=(1000, 1010),
                   colorbar=true, legend=false)

plot1 = plot(scatter1, xlabel="X Coordinate", ylabel="Y Coordinate",
             title="Density Distribution", colorbar_title="Density")
plot2 = plot(scatter2, xlabel="X Coordinate", ylabel="Y Coordinate",
             title="Density with 2x Smoothing Length", colorbar_title="Density")
plot3 = plot(scatter3, xlabel="X Coordinate", ylabel="Y Coordinate",
             title="Density with 0.5x Smoothing Length", colorbar_title="Density")

combined_plot = plot(plot1, plot2, plot3, layout=(1, 3), size=(1800, 600), margin=5mm)

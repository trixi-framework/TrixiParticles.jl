# Example for using interpolation
#######################################################################################
using TrixiParticles
# this needs to be commented out to use PythonPlot
using Plots
using Plots.PlotMeasures

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "rectangular_tank_2d.jl"),
              tspan=(0.0, 0.1))

# Interpolation parameters
interpolation_start = [0.0, 0.0]
interpolation_end = [1.0, 1.0]
resolution = 0.005

# We can interpolate a plane by providing the lower left and top right coordinates of a plane.
# Per default the same `smoothing_length` will be used as during the simulation.
original_plane = interpolate_plane_2d(interpolation_start, interpolation_end, resolution,
                                      semi,
                                      fluid_system, sol)
original_x = [point[1] for point in original_plane.coord]
original_y = [point[2] for point in original_plane.coord]
original_density = original_plane.density

# Plane with double smoothing length
# Using an higher `smoothing_length` will increase the amount of smoothing and will decrease
# the appearance of disturbances. At the same time it will also increase the distance at free surfaces
# at which the fluid is cut_off.
double_smoothing_plane = interpolate_plane_2d(interpolation_start, interpolation_end,
                                              resolution, semi, fluid_system, sol,
                                              smoothing_length=2.0 * smoothing_length)
double_x = [point[1] for point in double_smoothing_plane.coord]
double_y = [point[2] for point in double_smoothing_plane.coord]
double_density = double_smoothing_plane.density

# Plane with half smoothing length
# Using a lower `smoothing_length` will decrease the amount of smoothing and will increase
# the appearance of disturbances. At the same time the fluid will be cut off more accurately
# at free surfaces.
half_smoothing_plane = interpolate_plane_2d(interpolation_start, interpolation_end,
                                            resolution,
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

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "rectangular_tank_3d.jl"),
              tspan=(0.0, 0.1))

# Interpolation parameters
p1 = [0.0, 0.0, 0.0]
p2 = [1.0, 1.0, 0.1]
p3 = [1.0, 0.5, 0.2]
resolution = 0.025

# We can also interpolate a 3D plane but in this case we have to provide 3 points instead!
original_plane = interpolate_plane_3d(p1, p2, p3, resolution, semi,
                                      fluid_system, sol)
original_x = [point[1] for point in original_plane.coord]
original_y = [point[2] for point in original_plane.coord]
original_z = [point[3] for point in original_plane.coord]
original_density = original_plane.density

scatter_3d = scatter3d(original_x, original_y, original_z, marker_z=original_density,
                       color=:viridis, legend=false)

plot_3d = plot(scatter_3d, xlabel="X", ylabel="Y", zlabel="Z",
               title="3D Scatter Plot with Density Coloring")

# by ignoring the z coordinate we can also plot this into a 2D plane
scatter_3d_in_2d = scatter(original_x, original_y, zcolor=original_density,
                           marker=:circle, markersize=4,
                           markercolor=:viridis, markerstrokewidth=0, clim=(1000, 1010),
                           colorbar=true, legend=false)

plot_3d_in_2d = plot(scatter_3d_in_2d, xlabel="X", ylabel="Y", zlabel="Z",
                     title="3D in 2D Scatter Plot")

combined_plot = plot(plot1, plot2, plot3, plot_3d, plot_3d_in_2d, layout=(3, 2),
                     size=(1000, 1500), margin=5mm)

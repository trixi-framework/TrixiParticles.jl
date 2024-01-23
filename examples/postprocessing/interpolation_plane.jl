# Example for using interpolation
#######################################################################################
using TrixiParticles
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
                                      semi, fluid_system, sol)
original_x = [point[1] for point in original_plane.coord]
original_y = [point[2] for point in original_plane.coord]
original_density = original_plane.density

# Plane with double smoothing length
# Utilizing a higher `smoothing_length` in SPH interpolation increases the amount of smoothing,
# thereby reducing the visibility of disturbances. It also increases the distance
# from free surfaces where the fluid is cut off. This adjustment in `smoothing_length`
# can affect both the accuracy and smoothness of the interpolated results.
double_smoothing_plane = interpolate_plane_2d(interpolation_start, interpolation_end,
                                              resolution, semi, fluid_system, sol,
                                              smoothing_length=2.0 * smoothing_length)
double_x = [point[1] for point in double_smoothing_plane.coord]
double_y = [point[2] for point in double_smoothing_plane.coord]
double_density = double_smoothing_plane.density

# Plane with half smoothing length
# Employing a lower `smoothing_length` in SPH interpolation reduces the amount of smoothing,
# consequently increasing the visibility of disturbances. Simultaneously, it allows for a more
# precise cutoff of the fluid at free surfaces. This change in `smoothing_length` can impact the
# balance between the detail of disturbances captured and the precision of fluid representation near surfaces.
half_smoothing_plane = interpolate_plane_2d(interpolation_start, interpolation_end,
                                            resolution,
                                            semi, fluid_system, sol,
                                            smoothing_length=0.5 * smoothing_length)
half_x = [point[1] for point in half_smoothing_plane.coord]
half_y = [point[2] for point in half_smoothing_plane.coord]
half_density = half_smoothing_plane.density

scatter1 = scatter(original_x, original_y, zcolor=original_density, marker=:circle,
                   markersize=2, markercolor=:viridis, markerstrokewidth=0)
scatter2 = scatter(double_x, double_y, zcolor=double_density, marker=:circle, markersize=2,
                   markercolor=:viridis, markerstrokewidth=0)
scatter3 = scatter(half_x, half_y, zcolor=half_density, marker=:circle, markersize=2,
                   markercolor=:viridis, markerstrokewidth=0)

plot1 = plot(scatter1, xlabel="X Coordinate", ylabel="Y Coordinate",
             title="Density Distribution", colorbar_title="Density", ylim=(0.0, 1.0),
             legend=false, clim=(1000, 1010), colorbar=true)
plot2 = plot(scatter2, xlabel="X Coordinate", ylabel="Y Coordinate",
             title="Density with 2x Smoothing Length", colorbar_title="Density",
             ylim=(0.0, 1.0), legend=false, clim=(1000, 1010), colorbar=true)
plot3 = plot(scatter3, xlabel="X Coordinate", ylabel="Y Coordinate",
             title="Density with 0.5x Smoothing Length", colorbar_title="Density",
             ylim=(0.0, 1.0), legend=false, clim=(1000, 1010), colorbar=true)

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "rectangular_tank_3d.jl"),
              tspan=(0.0, 0.1), initial_fluid_size=(2.0, 1.0, 0.9),
              tank_size=(2.0, 1.0, 1.2))

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
                       color=:viridis, markerstrokewidth=0)

plot_3d = plot(scatter_3d, xlabel="X", ylabel="Y", zlabel="Z",
               title="3D Scatter Plot with Density Coloring", legend=false,
               clim=(1000, 1010), colorbar=false)

# by ignoring the z coordinate we can also plot this into a 2D plane
scatter_3d_in_2d = scatter(original_x, original_y, zcolor=original_density,
                           marker=:circle, markersize=4,
                           markercolor=:viridis, markerstrokewidth=0)

plot_3d_in_2d = plot(scatter_3d_in_2d, xlabel="X", ylabel="Y",
                     title="3D in 2D Scatter Plot", legend=false, clim=(1000, 1010),
                     colorbar=true)

combined_plot = plot(plot1, plot2, plot3, plot_3d, plot_3d_in_2d, layout=(3, 2),
                     size=(1000, 1500), margin=3mm)

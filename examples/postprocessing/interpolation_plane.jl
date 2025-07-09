# ==========================================================================================
# 2D and 3D Plane Interpolation Example
#
# This example demonstrates how to interpolate SPH particle data (e.g., pressure)
# onto a regular 2D or 3D plane using TrixiParticles.jl.
# A hydrostatic water column simulation is used as the base for generating particle data.
# ==========================================================================================

using TrixiParticles
using Plots
using Plots.PlotMeasures

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"),
              tspan=(0.0, 0.1))

# Interpolation parameters
interpolation_start = [0.0, 0.0]
interpolation_end = [1.0, 1.0]
resolution = 0.005

# We can interpolate a plane by providing the lower left and top right coordinates of a plane.
# Per default the same `smoothing_length` will be used as during the simulation.
original_plane = interpolate_plane_2d(interpolation_start, interpolation_end, resolution,
                                      semi, fluid_system, sol)
original_x = original_plane.point_coords[1, :]
original_y = original_plane.point_coords[2, :]
original_pressure = original_plane.pressure

# To export the interpolated plane as a VTI file, which can be read by tools like ParaView,
# run `interpolate_plane_2d_vtk` with the same arguments as above. When no other filename
# is specified with the kwarg `filename`, it will be exported to `out/plane.vti`.
interpolate_plane_2d_vtk(interpolation_start, interpolation_end, resolution,
                         semi, fluid_system, sol)

# Plane with double smoothing length.
# Utilizing a higher `smoothing_length` in SPH interpolation increases the amount of smoothing,
# thereby reducing the visibility of disturbances. It also increases the distance
# from free surfaces where the fluid is cut off. This adjustment in `smoothing_length`
# can affect both the accuracy and smoothness of the interpolated results.
double_smoothing_plane = interpolate_plane_2d(interpolation_start, interpolation_end,
                                              resolution, semi, fluid_system, sol,
                                              smoothing_length=2.0 * smoothing_length)
double_x = double_smoothing_plane.point_coords[1, :]
double_y = double_smoothing_plane.point_coords[2, :]
double_pressure = double_smoothing_plane.pressure

# Plane with half smoothing length.
# Employing a lower `smoothing_length` in SPH interpolation reduces the amount of smoothing,
# consequently increasing the visibility of disturbances. Simultaneously, it allows for a more
# precise cutoff of the fluid at free surfaces. This change in `smoothing_length` can impact the
# balance between the detail of disturbances captured and the precision of fluid representation near surfaces.
half_smoothing_plane = interpolate_plane_2d(interpolation_start, interpolation_end,
                                            resolution, semi, fluid_system, sol,
                                            smoothing_length=0.5 * smoothing_length)
half_x = half_smoothing_plane.point_coords[1, :]
half_y = half_smoothing_plane.point_coords[2, :]
half_pressure = half_smoothing_plane.pressure

scatter1 = Plots.scatter(original_x, original_y, zcolor=original_pressure, marker=:circle,
                         markersize=2, markercolor=:viridis, markerstrokewidth=0)
scatter2 = Plots.scatter(double_x, double_y, zcolor=double_pressure, marker=:circle,
                         markersize=2, markercolor=:viridis, markerstrokewidth=0)
scatter3 = Plots.scatter(half_x, half_y, zcolor=half_pressure, marker=:circle, markersize=2,
                         markercolor=:viridis, markerstrokewidth=0)

plot1 = Plots.plot(scatter1, xlabel="X Coordinate", ylabel="Y Coordinate",
                   title="Pressure Distribution", colorbar_title="Pressure",
                   ylim=(0.0, 1.0), legend=false, clim=(0, 9000), colorbar=true)
plot2 = Plots.plot(scatter2, xlabel="X Coordinate", ylabel="Y Coordinate",
                   title="Pressure with 2x Smoothing Length", colorbar_title="Pressure",
                   ylim=(0.0, 1.0), legend=false, clim=(0, 9000), colorbar=true)
plot3 = Plots.plot(scatter3, xlabel="X Coordinate", ylabel="Y Coordinate",
                   title="Pressure with 0.5x Smoothing Length", colorbar_title="Pressure",
                   ylim=(0.0, 1.0), legend=false, clim=(0, 9000), colorbar=true)

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "hydrostatic_water_column_3d.jl"),
              tspan=(0.0, 0.01), initial_fluid_size=(2.0, 1.0, 0.9),
              tank_size=(2.0, 1.0, 1.2))

# Interpolation parameters
p1 = [0.0, 0.0, 0.0]
p2 = [1.0, 1.0, 0.1]
p3 = [1.0, 0.5, 0.2]
resolution = 0.025

# We can also interpolate a 3D plane but in this case we have to provide 3 points instead!
original_plane = interpolate_plane_3d(p1, p2, p3, resolution, semi,
                                      fluid_system, sol)
original_x = original_plane.point_coords[1, :]
original_y = original_plane.point_coords[2, :]
original_z = original_plane.point_coords[3, :]
original_pressure = original_plane.pressure

scatter_3d = Plots.scatter3d(original_x, original_y, original_z, marker_z=original_pressure,
                             color=:viridis, markerstrokewidth=0)

plot_3d = Plots.plot(scatter_3d, xlabel="X", ylabel="Y", zlabel="Z",
                     title="3D Scatter Plot with Pressure Coloring", legend=false,
                     clim=(0, 9000), colorbar=false)

combined_plot = Plots.plot(plot1, plot2, plot3, plot_3d, layout=(2, 2),
                           size=(1000, 1500), margin=3mm)

# If we want to save planes at regular intervals, we can use the postprocessing callback.
# Note that the arguments `system, v_ode, u_ode, semi, t` are more powerful than the
# documented arguments `system, data, t`, allowing us to use interpolation (which requires
# a semidiscretization).
function save_interpolated_plane(system, v_ode, u_ode, semi, t)
    # Size of the patch to be interpolated
    interpolation_start = [0.0, 0.0]
    interpolation_end = [tank_size[1], tank_size[2]]

    # The resolution the plane is interpolated to. In this case twice the original resolution.
    resolution = 0.5 * fluid_particle_spacing

    file_id = ceil(Int, t * 10000)
    interpolate_plane_2d_vtk(interpolation_start, interpolation_end, resolution,
                             semi, system, v_ode, u_ode, filename="plane_$file_id.vti")
    return nothing
end

save_interpolation_cb = PostprocessCallback(; dt=0.1, write_file_interval=0,
                                            save_interpolated_plane)

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "dam_break_2d.jl"), tspan=(0.0, 0.2),
              extra_callback=save_interpolation_cb, fluid_particle_spacing=0.01)

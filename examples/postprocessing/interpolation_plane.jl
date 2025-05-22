# ==========================================================================================
# 2D and 3D Plane Interpolation Example
#
# This example demonstrates how to interpolate SPH particle data (e.g., pressure)
# onto a regular 2D or 3D plane using TrixiParticles.jl.#
# A hydrostatic water column simulation is used as the base for generating particle data.
# ==========================================================================================

using TrixiParticles
using Plots
using Plots.PlotMeasures # For plot margins

# ------------------------------------------------------------------------------
# Part 1: 2D Plane Interpolation and Visualization
# ------------------------------------------------------------------------------
# Run a short 2D hydrostatic water column simulation to get particle data.
# `sol`, `semi`, `fluid_system`, `smoothing_length`, `tank_size`, `fluid_particle_spacing`
# will be available from this included file.
println("Running 2D hydrostatic water column for plane interpolation...")
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"),
              tspan=(0.0, 0.1)) # Short simulation time

# Interpolation parameters for the 2D plane
plane_2d_start_coords = SVector(0.0, 0.0)   # Lower-left corner of the plane
plane_2d_end_coords = SVector(tank_size[1], tank_size[2]) # Upper-right corner (use tank_size from included sim)
interpolation_resolution_2d = 0.005        # Resolution of the interpolation grid

# --- Interpolate 2D plane with original smoothing length ---
# The `interpolate_plane_2d` function returns a struct containing coordinates and interpolated values.
# By default, it uses the `smoothing_length` from the `fluid_system`.
interpolated_plane_original_sl = interpolate_plane_2d(plane_2d_start_coords,
                                                      plane_2d_end_coords,
                                                      interpolation_resolution_2d,
                                                      semi, fluid_system, sol)
coords_x_orig = interpolated_plane_original_sl.point_coords[1, :]
coords_y_orig = interpolated_plane_original_sl.point_coords[2, :]
pressure_orig = interpolated_plane_original_sl.pressure

# Export the interpolated plane to a VTI file for ParaView.
# If `filename` is not specified, it defaults to "out/plane.vti".
interpolate_plane_2d_vtk(plane_2d_start_coords, plane_2d_end_coords,
                         interpolation_resolution_2d,
                         semi, fluid_system, sol, filename="out/interpolated_plane_2d_orig_sl.vti")
println("Exported 2D plane with original smoothing length to out/interpolated_plane_2d_orig_sl.vti")

# --- Interpolate 2D plane with double smoothing length ---
# Increasing the smoothing length for interpolation results in more smoothing of the
# interpolated field, potentially obscuring fine details but reducing noise.
# It also affects how the fluid is "seen" near free surfaces.
interpolated_plane_double_sl = interpolate_plane_2d(plane_2d_start_coords,
                                                     plane_2d_end_coords,
                                                     interpolation_resolution_2d,
                                                     semi, fluid_system, sol,
                                                     smoothing_length=2.0 * smoothing_length) # Override SL
coords_x_double = interpolated_plane_double_sl.point_coords[1, :]
coords_y_double = interpolated_plane_double_sl.point_coords[2, :]
pressure_double = interpolated_plane_double_sl.pressure

# --- Interpolate 2D plane with half smoothing length ---
# Decreasing the smoothing length reduces smoothing, making disturbances more visible
# and providing a sharper cutoff at free surfaces, but can increase noise.
interpolated_plane_half_sl = interpolate_plane_2d(plane_2d_start_coords,
                                                  plane_2d_end_coords,
                                                  interpolation_resolution_2d,
                                                  semi, fluid_system, sol,
                                                  smoothing_length=0.5 * smoothing_length) # Override SL
coords_x_half = interpolated_plane_half_sl.point_coords[1, :]
coords_y_half = interpolated_plane_half_sl.point_coords[2, :]
pressure_half = interpolated_plane_half_sl.pressure

# --- Plotting the 2D interpolated planes ---
# Pressure limits for colorbar consistency
pressure_clim = (0, maximum(pressure_orig) * 1.1) # Based on original SL, slightly above max

scatter_orig = Plots.scatter(coords_x_orig, coords_y_orig, zcolor=pressure_orig,
                             marker=:circle, markersize=2, markercolor=:viridis,
                             markerstrokewidth=0, aspect_ratio=:equal)
plot_orig = Plots.plot(scatter_orig, xlabel="X-coordinate", ylabel="Y-coordinate",
                       title="Pressure (Original SL)", colorbar_title="Pressure",
                       legend=false, clim=pressure_clim, colorbar=true)

scatter_double = Plots.scatter(coords_x_double, coords_y_double, zcolor=pressure_double,
                               marker=:circle, markersize=2, markercolor=:viridis,
                               markerstrokewidth=0, aspect_ratio=:equal)
plot_double = Plots.plot(scatter_double, xlabel="X-coordinate", ylabel="Y-coordinate",
                         title="Pressure (2x SL)", colorbar_title="Pressure",
                         legend=false, clim=pressure_clim, colorbar=true)

scatter_half = Plots.scatter(coords_x_half, coords_y_half, zcolor=pressure_half,
                             marker=:circle, markersize=2, markercolor=:viridis,
                             markerstrokewidth=0, aspect_ratio=:equal)
plot_half = Plots.plot(scatter_half, xlabel="X-coordinate", ylabel="Y-coordinate",
                       title="Pressure (0.5x SL)", colorbar_title="Pressure",
                       legend=false, clim=pressure_clim, colorbar=true)

# ------------------------------------------------------------------------------
# Part 2: 3D Plane Interpolation and Visualization
# ------------------------------------------------------------------------------
# Run a short 3D hydrostatic water column simulation.
# This will redefine `sol`, `semi`, `fluid_system`, etc.
println("\nRunning 3D hydrostatic water column for plane interpolation...")
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "hydrostatic_water_column_3d.jl"),
              tspan=(0.0, 0.01), # Very short, just to get initial state
              initial_fluid_size=(2.0, 1.0, 0.9), # Use default from example
              tank_size=(2.0, 1.0, 1.2))       # Use default from example

# Interpolation parameters for an arbitrary 3D plane defined by three points.
point1_3d = SVector(0.0, 0.0, 0.0)
point2_3d = SVector(tank_size[1], tank_size[2], 0.1) # Example points spanning the tank
point3_3d = SVector(tank_size[1], 0.5 * tank_size[2], 0.2)
interpolation_resolution_3d = 0.025

interpolated_plane_3d = interpolate_plane_3d(point1_3d, point2_3d, point3_3d,
                                             interpolation_resolution_3d,
                                             semi, fluid_system, sol)
coords_x_3d = interpolated_plane_3d.point_coords[1, :]
coords_y_3d = interpolated_plane_3d.point_coords[2, :]
coords_z_3d = interpolated_plane_3d.point_coords[3, :]
pressure_3d = interpolated_plane_3d.pressure

# --- Plotting the 3D interpolated plane ---
pressure_clim_3d = (0, maximum(pressure_3d) * 1.1) # Similar clim logic for 3D

scatter_3d_plot = Plots.scatter3d(coords_x_3d, coords_y_3d, coords_z_3d,
                                  marker_z=pressure_3d, color=:viridis,
                                  markerstrokewidth=0, markersize=2)
plot_3d_interpolated = Plots.plot(scatter_3d_plot, xlabel="X", ylabel="Y", zlabel="Z",
                                  title="3D Interpolated Plane Pressure", legend=false,
                                  clim=pressure_clim_3d, colorbar=true, camera=(30, 30))

# --- Combine all plots ---
println("Displaying combined plots...")
combined_plot_figure = Plots.plot(plot_orig, plot_double, plot_half, plot_3d_interpolated,
                                  layout=(2, 2), size=(1200, 1000), margin=5mm)
display(combined_plot_figure) # Ensure the plot is displayed

# ------------------------------------------------------------------------------
# Part 3: Saving Interpolated Planes Periodically using PostprocessCallback
# ------------------------------------------------------------------------------
# Define a function to be called by the PostprocessCallback.
# This function will interpolate a 2D plane and save it as a VTK file.
# The arguments `system, v_ode, u_ode, semi, t` provide access to the current
# simulation state needed for interpolation.
function save_interpolated_plane_callback_function(system, v_ode, u_ode, semi, t)
    # Define plane for interpolation (e.g., entire tank domain)
    # `tank_size` and `fluid_particle_spacing` need to be in scope or passed.
    # Here, we assume they are available from the `trixi_include` that sets up the callback.
    callback_plane_start = SVector(0.0, 0.0)
    # Use `semi.initial_condition.fluid_system.configuration.tank_size` or similar robust way if needed
    callback_plane_end = SVector(semi.systems[1].initial_coordinates[1,end], semi.systems[1].initial_coordinates[2,end]) # Heuristic for tank size

    # Resolution for interpolation (e.g., twice the fluid particle spacing)
    callback_resolution = 0.5 * semi.systems[1].reference_particle_spacing # Assuming fluid_system is first

    # Generate a unique filename based on time
    # Ensure "out/" directory exists if not created by SolutionSavingCallback
    mkpath("out")
    filename_vtk = "out/interpolated_plane_t$(round(Int, t*1000)).vti"

    println("Interpolating and saving plane to $filename_vtk at t=$t")
    interpolate_plane_2d_vtk(callback_plane_start, callback_plane_end, callback_resolution,
                             semi, system, v_ode, u_ode; filename=filename_vtk) # Pass filename kwarg

    return nothing # Callback functions usually return nothing
end

# Create a PostprocessCallback to execute the save function at specified intervals.
# `dt=0.1` means the callback function is called every 0.1 simulation time units.
# `write_file_interval=0` means the callback itself doesn't write a summary CSV/JSON file.
save_interpolation_postprocess_cb = PostprocessCallback(; dt=0.1, write_file_interval=0,
                                                        user_defined_callback=save_interpolated_plane_callback_function)

# Run a 2D dam break simulation with the PostprocessCallback.
println("\nRunning 2D dam break with periodic plane interpolation saving...")
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              tspan=(0.0, 0.2), # Short simulation
              extra_callback=save_interpolation_postprocess_cb,
              fluid_particle_spacing=0.01) # Use a finer resolution for demo

println("Periodic plane interpolation example finished. Check 'out/' directory for VTI files.")

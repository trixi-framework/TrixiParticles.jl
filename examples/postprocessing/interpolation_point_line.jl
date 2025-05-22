# ==========================================================================================
# Point and Line Interpolation Example
#
# This example demonstrates how to interpolate SPH particle data (e.g., density, pressure)
# at specific points or along a line using TrixiParticles.jl.
# A hydrostatic water column simulation is used as the base for generating particle data.
# ==========================================================================================

using TrixiParticles
using Plots # For plotting line interpolation results

# ------------------------------------------------------------------------------
# Setup: Run a Base Simulation
# ------------------------------------------------------------------------------
# Run a 2D hydrostatic water column simulation to get particle data.
# `sol`, `semi`, `fluid_system`, `smoothing_length`, `tank_size`, `fluid_particle_spacing`
# will be available from this included file.
println("Running 2D hydrostatic water column for point/line interpolation...")
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"))

# ------------------------------------------------------------------------------
# Part 1: Point Interpolation
# ------------------------------------------------------------------------------
# Define a point for interpolation (e.g., center of the tank, slightly above the bottom).
point_x_coord = tank_size[1] / 2
point_y_coord_inside = 0.01 # Well within the fluid
point_for_interpolation = SVector(point_x_coord, point_y_coord_inside)

# --- Interpolate at a single point with original smoothing length ---
# `interpolate_points` takes a 2xN matrix of points (each column is a point).
# It returns a struct containing arrays of interpolated properties.
interpolated_data_single_point = interpolate_points(reshape(point_for_interpolation, 2, 1),
                                                    semi, fluid_system, sol)
println("\nInterpolated data at $point_for_interpolation (original SL):")
println("  Density: ", interpolated_data_single_point.density[1])
println("  Pressure: ", interpolated_data_single_point.pressure[1])

# --- Interpolate at the same point with increased smoothing length ---
# This generally results in a smoother (more averaged) value.
interpolated_data_double_sl = interpolate_points(reshape(point_for_interpolation, 2, 1),
                                                 semi, fluid_system, sol,
                                                 smoothing_length=2.0 * smoothing_length)
println("\nInterpolated data at $point_for_interpolation (2x SL):")
println("  Density: ", interpolated_data_double_sl.density[1])
println("  Pressure: ", interpolated_data_double_sl.pressure[1])

# --- Interpolation at and outside the fluid domain ---
point_on_boundary = SVector(point_x_coord, 0.0) # Exactly on the bottom boundary
point_outside_domain = SVector(point_x_coord, -0.01) # Slightly below the bottom

data_on_boundary = interpolate_points(reshape(point_on_boundary, 2, 1),
                                      semi, fluid_system, sol)
println("\nInterpolated data at boundary point $point_on_boundary:")
println("  Density: ", data_on_boundary.density[1]) # May get a value due to kernel support

data_outside_domain = interpolate_points(reshape(point_outside_domain, 2, 1),
                                         semi, fluid_system, sol)
println("\nInterpolated data outside domain $point_outside_domain:")
println("  Density: ", data_outside_domain.density[1]) # Should be NaN or zero

# --- Interpolate at multiple points simultaneously ---
points_matrix = [point_x_coord        point_x_coord point_on_boundary[1] point_outside_domain[1] point_outside_domain[1]-0.04;
                 point_y_coord_inside 0.1           point_on_boundary[2] point_outside_domain[2] point_outside_domain[2]-0.04]

interpolated_data_multiple = interpolate_points(points_matrix, semi, fluid_system, sol)
println("\nInterpolated data for multiple points:")
for i in 1:size(points_matrix, 2)
    println("  Point $(points_matrix[:, i]): Density = $(interpolated_data_multiple.density[i]), Pressure = $(interpolated_data_multiple.pressure[i])")
end

# ------------------------------------------------------------------------------
# Part 2: Line Interpolation
# ------------------------------------------------------------------------------
# Define a vertical line for interpolation.
num_interpolation_points_on_line = 20 # Number of points along the line
line_start_point = SVector(point_x_coord, -fluid_particle_spacing) # Start slightly below fluid
line_end_point = SVector(point_x_coord, tank_size[2])             # End at top of tank

# --- Interpolate along the line, including the endpoint ---
interpolated_line_data_with_endpoint = interpolate_line(line_start_point, line_end_point,
                                                        num_interpolation_points_on_line,
                                                        semi, fluid_system, sol,
                                                        endpoint=true) # Default is true

# --- Interpolate along the line, excluding the endpoint ---
# The `endpoint=false` option means the line segment does not include `line_end_point`.
interpolated_line_data_no_endpoint = interpolate_line(line_start_point, line_end_point,
                                                      num_interpolation_points_on_line,
                                                      semi, fluid_system, sol,
                                                      endpoint=false)

# Extract y-coordinates (distance along the line, effectively "wall distance" from bottom)
# and density for plotting.
y_coords_with_endpoint = interpolated_line_data_with_endpoint.point_coords[2, :]
density_with_endpoint = interpolated_line_data_with_endpoint.density

y_coords_no_endpoint = interpolated_line_data_no_endpoint.point_coords[2, :]
density_no_endpoint = interpolated_line_data_no_endpoint.density

# Replace NaNs with zeros for cleaner plotting (NaNs occur outside fluid support).
replace!(density_with_endpoint, NaN => 0.0)
replace!(density_no_endpoint, NaN => 0.0)

# --- Plotting the line interpolation results ---
# Using Plots.jl (default)
# To use PythonPlot (matplotlib backend), uncomment the relevant lines below and comment out Plots.jl usage.
# using PythonPlot
# figure()
# plot(y_coords_with_endpoint, density_with_endpoint, marker="o", linestyle="-", label="With Endpoint")
# plot(y_coords_no_endpoint, density_no_endpoint, marker="x", linestyle="--", label="Without Endpoint")
# xlabel("Y-coordinate (Distance from bottom)")
# ylabel("Interpolated Density")
# title("Density Interpolation Along a Vertical Line")
# legend()
# grid(true)
# PythonPlot.show() # Use PythonPlot.show() for matplotlib

# Plots.jl plotting
plot_line_interpolation = Plots.plot(y_coords_with_endpoint, density_with_endpoint,
                                     marker=:circle, color=:blue,
                                     label="With Endpoint", linewidth=1.5)
Plots.plot!(plot_line_interpolation, y_coords_no_endpoint, density_no_endpoint,
            marker=:xcross, color=:orange, linestyle=:dash,
            label="Without Endpoint", linewidth=1.5)

Plots.xlabel!(plot_line_interpolation, "Y-coordinate (Distance from bottom)")
Plots.ylabel!(plot_line_interpolation, "Interpolated Density")
Plots.title!(plot_line_interpolation, "Density Interpolation Along a Vertical Line")
Plots.legend!(plot_line_interpolation, :topright)
Plots.plot!(plot_line_interpolation, framestyle=:box, size=(800, 600), dpi=150)

println("\nDisplaying line interpolation plot...")
display(plot_line_interpolation)

println("\nPoint and line interpolation example finished.")

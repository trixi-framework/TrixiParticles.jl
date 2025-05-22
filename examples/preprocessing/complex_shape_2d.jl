# ==========================================================================================
# 2D Complex Shape Sampling and Winding Number Visualization
#
# This example demonstrates how to:
# 1. Load a 2D geometry from an ASCII file (e.g., a curve).
# 2. Sample particles within this complex geometry using the `ComplexShape` functionality.
# 3. Utilize the Winding Number algorithm to determine if points are inside or outside.
# 4. Visualize the sampled particles and the winding number field.
#
# The example uses an "inverted_open_curve" geometry, where standard inside/outside
# definitions might be ambiguous without a robust point-in-polygon test like winding numbers.
# ==========================================================================================

using TrixiParticles
using Plots # For visualizing the winding number field
using Plots.PlotMeasures # For plot margins, if needed

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------
# Particle spacing for sampling the complex shape
particle_spacing = 0.05 # meters

# Geometry file details
geometry_filename_stem = "inverted_open_curve"
# Assuming the data file is in "examples/preprocessing/data/" relative to TrixiParticles.jl root
geometry_file_path = joinpath(pkgdir(TrixiParticles), "examples", "preprocessing", "data",
                              geometry_filename_stem * ".asc")

# Parameters for the Winding Number algorithm
# `winding_number_factor` helps classify points near the boundary.
# `hierarchical_winding` uses a multi-level approach for robustness and efficiency.
winding_number_threshold_factor = 0.4 # Points with winding number > factor are considered inside
use_hierarchical_winding = true

# ------------------------------------------------------------------------------
# Load Geometry and Sample Particles
# ------------------------------------------------------------------------------
# Load the 2D geometry from the specified file.
# `geometry` will be a `BoundaryPath` object or similar.
println("Loading 2D geometry from: $geometry_file_path")
geometry = load_geometry(geometry_file_path)

# Optional: Export the loaded geometry to a VTK file for inspection.
# This helps verify that the geometry was loaded correctly.
output_vtk_geometry_filename = "out/$(geometry_filename_stem)_boundary.vtk"
trixi2vtk(geometry, filename=output_vtk_geometry_filename)
println("Exported loaded geometry to $output_vtk_geometry_filename")

# Define the point-in-geometry algorithm using Winding Numbers.
# This algorithm determines which grid points (for sampling) fall inside the geometry.
point_in_geometry_test = WindingNumberJacobson(geometry=geometry,
                                               winding_number_factor=winding_number_threshold_factor,
                                               hierarchical_winding=use_hierarchical_winding)

# Sample particles within the complex shape.
# `ComplexShape` generates an `InitialCondition` object containing particle positions and properties.
# `store_winding_number=true` stores the calculated winding number for each sampled point.
println("Sampling particles within the complex shape...")
sampled_shape_data = ComplexShape(geometry;
                                  particle_spacing=particle_spacing,
                                  density=1.0, # Dummy density for visualization
                                  store_winding_number=true,
                                  point_in_geometry_algorithm=point_in_geometry_test)

# The actual particle data is in `sampled_shape_data.initial_condition`.
initial_condition_particles = sampled_shape_data.initial_condition

# Export the sampled particles (initial condition) to a VTK file.
output_vtk_particles_filename = "out/$(geometry_filename_stem)_sampled_particles.vtp"
trixi2vtk(initial_condition_particles, filename=output_vtk_particles_filename)
println("Exported sampled particles to $output_vtk_particles_filename")

# ------------------------------------------------------------------------------
# Visualize Winding Numbers (Optional)
# ------------------------------------------------------------------------------
# `sampled_shape_data.grid` contains all grid points considered during sampling.
# `sampled_shape_data.winding_numbers` contains the winding number for each of these grid points.

# Extract coordinates of all grid points and their winding numbers.
grid_point_coordinates = stack(sampled_shape_data.grid) # Converts vector of SVectors to matrix
grid_winding_numbers = sampled_shape_data.winding_numbers

# Create an `InitialCondition` object for plotting all grid points colored by winding number.
# This helps visualize the winding number field.
grid_visualization_ic = InitialCondition(coordinates=grid_point_coordinates,
                                         density=1.0, # Dummy density
                                         particle_spacing=particle_spacing) # For marker size in plot

println("Plotting the winding number field for all considered grid points...")
winding_number_plot = plot(grid_visualization_ic,
                           zcolor=grid_winding_numbers, # Color by winding number
                           markersize=2, markerstrokewidth=0,
                           colorbar_title="Winding Number",
                           title="Winding Number Field of '$geometry_filename_stem'",
                           aspect_ratio=:equal)
# Overlay the original geometry boundary for context
plot!(winding_number_plot, geometry, linecolor=:black, linewidth=1.5, label="Geometry Boundary")

display(winding_number_plot)
println("Complex shape 2D example finished. Check 'out/' directory for VTK files and plot window.")

# Additional advanced visualization (commented out, requires specific understanding):
# - `sampled_shape_data.signed_distance_field`: If `create_signed_distance_field=true` was used.
#   trixi2vtk(sampled_shape_data.signed_distance_field)
# - Winding numbers for only the *sampled* (inside) particles:
#   To get this, one would filter `sampled_shape_data.grid` based on where particles were placed
#   in `initial_condition_particles` and then map the winding numbers.
#   The current `sampled_shape_data.winding_numbers` is for all grid points.

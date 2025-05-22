# ==========================================================================================
# 3D Complex Shape Sampling (e.g., from STL)
#
# This example demonstrates how to:
# 1. Load a 3D geometry from an STL file (e.g., a sphere).
# 2. Sample particles either as a fluid volume within this geometry or as a boundary layer.
# 3. Optionally create a Signed Distance Field (SDF) from the geometry.
# 4. Export the results to VTK files for visualization.
#
# The Winding Number algorithm is typically used for robust point-in-volume tests for 3D.
# ==========================================================================================

using TrixiParticles

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------
# Particle spacing for sampling
particle_spacing = 0.05 # meters

# Geometry file details
geometry_filename_stem = "sphere" # Example: a sphere STL
geometry_file_path = joinpath(pkgdir(TrixiParticles), "examples", "preprocessing", "data",
                              geometry_filename_stem * ".stl")

# Parameters for the Winding Number algorithm (if used for fluid sampling)
# For 3D, the winding number concept is more complex. A common threshold is 0.5.
# `hierarchical_winding` is generally recommended for 3D.
winding_number_threshold_factor_3d = 0.5 # Typical threshold for 3D closed surfaces
use_hierarchical_winding_3d = true

# Sampling options for ComplexShape
sample_as_fluid_volume = true  # If true, sample particles filling the geometry interior.
sample_as_boundary_layer = false # If true, sample particles on/near the geometry surface.
# Note: `sample_boundary` in `ComplexShape` refers to sampling boundary *particles*,
# not just evaluating the geometry boundary.
# If `sample_as_fluid_volume` is true, `sample_boundary=false` is common.
# If only a boundary layer is needed, `sample_as_fluid_volume=false` and `sample_boundary=true` might be used.

# Thickness of the boundary layer if `sample_as_boundary_layer` or `sample_boundary=true` is active.
boundary_sampling_thickness = 5 * particle_spacing

# Option to create a Signed Distance Field (SDF) from the geometry.
generate_signed_distance_field = true

# ------------------------------------------------------------------------------
# Load Geometry and Sample Particles
# ------------------------------------------------------------------------------
println("Loading 3D geometry from: $geometry_file_path")
geometry = load_geometry(geometry_file_path) # `geometry` will be a `TriangulatedSurface` or similar.

# Optional: Export the loaded geometry for verification.
output_vtk_geometry_filename_3d = "out/$(geometry_filename_stem)_3d_boundary.vtp"
trixi2vtk(geometry, filename=output_vtk_geometry_filename_3d)
println("Exported loaded 3D geometry to $output_vtk_geometry_filename_3d")

# Define the point-in-geometry algorithm (used if `sample_as_fluid_volume=true`).
point_in_geometry_test_3d = WindingNumberJacobson(geometry=geometry,
                                                  winding_number_factor=winding_number_threshold_factor_3d,
                                                  hierarchical_winding=use_hierarchical_winding_3d)

# Sample particles using ComplexShape.
# The behavior depends on `sample_boundary` and how the algorithm is used.
# Original example implies fluid sampling (`sample_boundary=false`).
println("Sampling particles for the 3D complex shape...")
sampled_shape_data_3d = ComplexShape(geometry;
                                     particle_spacing=particle_spacing,
                                     density=1.0, # Dummy density
                                     # `boundary_thickness` is used if `sample_boundary=true`
                                     boundary_thickness=boundary_sampling_thickness,
                                     # `sample_boundary=false` means we primarily want interior/fluid particles
                                     sample_boundary=sample_as_boundary_layer,
                                     # Algorithm to determine if grid points are inside (for fluid)
                                     point_in_geometry_algorithm=point_in_geometry_test_3d,
                                     create_signed_distance_field=generate_signed_distance_field)

# Export the sampled fluid particles (initial condition).
# These are the particles inside the geometry if `sample_boundary=false` was effectively used
# for fluid volume sampling.
output_vtk_fluid_particles_3d = "out/$(geometry_filename_stem)_3d_fluid_particles.vtp"
trixi2vtk(sampled_shape_data_3d.initial_condition, filename=output_vtk_fluid_particles_3d)
println("Exported sampled 3D fluid particles to $output_vtk_fluid_particles_3d")

# If a boundary layer was sampled (`sample_boundary=true`), it might be in a different part
# of `sampled_shape_data_3d` or combined, depending on `ComplexShape` internal logic.
# The original example implies the main output `initial_condition` is the fluid.

# Export the Signed Distance Field (SDF) if generated.
# The SDF is a grid where each point stores its shortest distance to the geometry surface.
if generate_signed_distance_field && !isnothing(sampled_shape_data_3d.signed_distance_field)
    output_vtk_sdf_3d = "out/$(geometry_filename_stem)_3d_sdf.vti" # SDF is usually a grid (VTI)
    trixi2vtk(sampled_shape_data_3d.signed_distance_field, filename=output_vtk_sdf_3d)
    println("Exported 3D Signed Distance Field to $output_vtk_sdf_3d")
else
    println("Signed Distance Field was not generated or is not available.")
end

println("Complex shape 3D example finished. Check 'out/' directory for VTK files.")

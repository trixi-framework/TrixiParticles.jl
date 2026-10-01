# ==========================================================================================
# 3D Complex Shape Sampling (e.g., from STL)
#
# This example demonstrates how to:
# 1. Load a 3D geometry from an STL file (e.g., a sphere).
# 2. Sample particles as a fluid volume within this geometry and as a boundary layer around it.
# 3. Use a Signed Distance Field (SDF) to sample the boundary layer.
# 4. Export the results to VTK files for visualization.
#
# The Winding Number algorithm is typically used for robust point-in-volume tests in 3D.
# ==========================================================================================

using TrixiParticles

particle_spacing = 0.05

filename = "sphere"
file = joinpath("examples", "preprocessing", "data", filename * ".stl")

geometry = load_geometry(file)

point_in_geometry_algorithm = WindingNumberJacobson(; geometry)

# Returns `InitialCondition`
shape_sampled = ComplexShape(geometry; particle_spacing, density=1.0,
                             point_in_geometry_algorithm)

# Boundary particles can be sampled in a layer of thickness `boundary_thickness`
# around the geometry by using a signed distance field (SDF).
boundary_thickness = 5 * particle_spacing
signed_distance_field = SignedDistanceField(geometry, particle_spacing;
                                            use_for_boundary_packing=true,
                                            max_signed_distance=boundary_thickness)
boundary_sampled = sample_boundary(signed_distance_field; boundary_density=1.0,
                                   boundary_thickness)

trixi2vtk(shape_sampled)
trixi2vtk(boundary_sampled, filename="boundary_sampled")

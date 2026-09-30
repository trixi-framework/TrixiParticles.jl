# ==========================================================================================
# 3D Complex Shape Sampling (e.g., from STL)
#
# This example demonstrates how to:
# 1. Load a 3D geometry from an STL file (e.g., a sphere).
# 2. Sample particles as a fluid volume within this geometry.
# 3. Create a Signed Distance Field (SDF) from the geometry.
# 4. Export the results to VTK files for visualization.
#
# The Winding Number algorithm is typically used for robust point-in-volume tests in 3D.
# ==========================================================================================

using TrixiParticles

particle_spacing = 0.05

filename = "sphere"
file = pkgdir(TrixiParticles, "examples", "preprocessing", "data", filename * ".stl")

geometry = load_geometry(file)

point_in_geometry_algorithm = WindingNumberJacobson(; geometry)

# Returns `InitialCondition`
shape_sampled = ComplexShape(geometry; particle_spacing, density=1.0,
                             point_in_geometry_algorithm)

signed_distance_field = SignedDistanceField(geometry, particle_spacing)

trixi2vtk(shape_sampled)
trixi2vtk(signed_distance_field)

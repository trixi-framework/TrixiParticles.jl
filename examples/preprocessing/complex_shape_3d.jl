# ==========================================================================================
# 3D Complex Shape Sampling (e.g., from STL)
#
# This example demonstrates how to:
# 1. Load a 3D geometry from an STL file (e.g., a sphere).
# 2. Sample particles either as a fluid volume within this geometry or as a boundary layer.
# 3. Optionally create a Signed Distance Field (SDF) from the geometry.
# 4. Export the results to VTK files for visualization.
#
# The Winding Number algorithm is typically used for robust point-in-volume tests in 3D.
# ==========================================================================================

using TrixiParticles

particle_spacing = 0.05

filename = "sphere"
file = joinpath("examples", "preprocessing", "data", filename * ".stl")

geometry = load_geometry(file; parallelization_backend=false,
                         element_type=typeof(particle_spacing))

trixi2vtk(geometry)

# Returns `InitialCondition`
shape_sampled = ComplexShape(geometry; particle_spacing, density=1.0,
                             point_in_geometry_algorithm=WindingNumberJacobson(geometry))

trixi2vtk(shape_sampled)

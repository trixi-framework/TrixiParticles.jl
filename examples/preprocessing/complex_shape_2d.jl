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
using Plots

particle_spacing = 0.1

filename = "inverted_open_curve"
file = joinpath(examples_dir(), "preprocessing", "data", filename * ".asc")

geometry = load_geometry(file; parallelization_backend=PolyesterBackend(),
                         element_type=typeof(particle_spacing))

trixi2vtk(geometry)

point_in_geometry_algorithm = WindingNumberJacobson(geometry;
                                                    winding=HierarchicalWinding(geometry),
                                                    winding_number_factor=0.4,
                                                    store_winding_number=true)

# Returns `InitialCondition`
shape_sampled = ComplexShape(geometry; particle_spacing, density=1.0,
                             point_in_geometry_algorithm)

trixi2vtk(shape_sampled)

# Plot the winding number field
coordinates = stack(point_in_geometry_algorithm.cache.grid)
plot(InitialCondition(; coordinates, density=1.0, particle_spacing),
     zcolor=point_in_geometry_algorithm.cache.winding_numbers)

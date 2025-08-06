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

particle_spacing = 0.05

filename = "inverted_open_curve"
file = joinpath("examples", "preprocessing", "data", filename * ".asc")

geometry = load_geometry(file)

trixi2vtk(geometry)

point_in_geometry_algorithm = WindingNumberJacobson(; geometry,
                                                    winding_number_factor=0.4,
                                                    hierarchical_winding=true)

# Returns `InitialCondition`
shape_sampled = ComplexShape(geometry; particle_spacing, density=1.0,
                             store_winding_number=true,
                             point_in_geometry_algorithm)

trixi2vtk(shape_sampled.initial_condition)

coordinates = stack(shape_sampled.grid)
# trixi2vtk(shape_sampled.signed_distance_field)
# trixi2vtk(coordinates, w=shape_sampled.winding_numbers)

# Plot the winding number field
plot(InitialCondition(; coordinates, density=1.0, particle_spacing),
     zcolor=shape_sampled.winding_numbers)

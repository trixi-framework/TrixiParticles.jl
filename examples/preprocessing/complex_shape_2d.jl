using TrixiParticles
using Plots

particle_spacing = 0.08

filename = "box"
file = joinpath("examples", "preprocessing", "data", filename * ".asc")

geometry = load_geometry(file)

filename_diff = "circle"
file_diff = joinpath("examples", "preprocessing", "data", filename_diff * ".asc")

geometry_diff = load_geometry(file_diff)

setdiff!(geometry, geometry_diff)

trixi2vtk(geometry)

point_in_geometry_algorithm = WindingNumberJacobson(; geometry,
                                                    winding_number_factor=0.4,
                                                    hierarchical_winding=true)

# Returns `InitialCondition`
shape_sampled = ComplexShape(geometry; particle_spacing, density=1.0,
                             grid_offset=0.1particle_spacing,
                             store_winding_number=true, point_in_geometry_algorithm)

trixi2vtk(shape_sampled.initial_condition)

# trixi2vtk(shape_sampled.grid, w=shape_sampled.winding_numbers)

# Plot the winding number field
plot(InitialCondition(; coordinates=shape_sampled.grid, density=1.0, particle_spacing),
     zcolor=shape_sampled.winding_numbers)

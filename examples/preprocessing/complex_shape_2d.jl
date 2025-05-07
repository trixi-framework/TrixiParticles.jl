using TrixiParticles
using Plots

particle_spacing = 0.05

filename = "inverted_open_curve"
file = joinpath("examples", "preprocessing", "data", filename * ".asc")

geometry = load_geometry(file; parallelization_backend=true,
                         element_type=typeof(particle_spacing))

trixi2vtk(geometry)

point_in_geometry_algorithm = WindingNumberJacobson(; geometry,
                                                    winding_number_factor=0.4,
                                                    store_winding_number=true,
                                                    hierarchical_winding=true)

# Returns `InitialCondition`
shape_sampled = ComplexShape(geometry; particle_spacing, density=1.0,
                             point_in_geometry_algorithm)

trixi2vtk(shape_sampled)

coordinates = stack(shape_sampled.grid)
# trixi2vtk(shape_sampled.signed_distance_field)
# trixi2vtk(coordinates, w=shape_sampled.winding_numbers)

# Plot the winding number field
plot(InitialCondition(; coordinates, density=1.0, particle_spacing),
     zcolor=point_in_geometry_algorithm.cache.winding_numbers)

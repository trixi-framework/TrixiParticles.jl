using TrixiParticles

particle_spacing = 0.05

file = "hexagon"
filename = joinpath("examples", "preprocessing", file * ".asc")

geometry = load_geometry(filename)

trixi2vtk(geometry)

point_in_geometry_algorithm = WindingNumberJacobson(; geometry,
                                                    #winding_number_factor=0.4,
                                                    hierarchical_winding=true)

# Returns `InitialCondition`.
shape_sampled = ComplexShape(geometry; particle_spacing, density=1.0,
                             store_winding_number=true,
                             point_in_geometry_algorithm)

trixi2vtk(shape_sampled.initial_condition)
trixi2vtk(shape_sampled.grid, w=shape_sampled.winding_numbers)

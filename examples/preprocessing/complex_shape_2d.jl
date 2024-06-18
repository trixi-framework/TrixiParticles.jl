using TrixiParticles

particle_spacing = 0.05

filename = joinpath("examples", "preprocessing", "inverted_open_curve.asc")

# Returns `Shape`
shape = load_shape(filename)

# Returns `InitialCondition`.
shape_sampled = ComplexShape(shape; particle_spacing, density=1.0,
                             winding_number_factor=0.45, store_winding_number=true)

trixi2vtk(shape_sampled.grid, w=shape_sampled.winding_numbers, filename="grid")
trixi2vtk(shape_sampled.initial_condition.coordinates, filename="coords");

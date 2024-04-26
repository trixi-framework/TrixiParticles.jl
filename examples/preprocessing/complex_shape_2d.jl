using TrixiParticles

particle_spacing = 0.5

filename = joinpath("out_preprocessing", "hexagon.asc")

points = TrixiParticles.read_in_2d(; filename)

# Returns `Shape`
shape = load_shape(filename)

# Returns `InitialCondition`.
shape_sampled = ComplexShape(shape; particle_spacing, density=1.0)


trixi2vtk(points, filename="points")
trixi2vtk(shape_sampled.coordinates, filename="coords");

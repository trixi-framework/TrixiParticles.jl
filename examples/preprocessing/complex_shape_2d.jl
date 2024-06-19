using TrixiParticles

particle_spacing = 0.05

filename = joinpath("examples", "preprocessing", "hexagon.asc")

# Returns `Shape`
shape = load_shape(filename)

# Returns `InitialCondition`.
shape_sampled = ComplexShape(shape; particle_spacing, density=1.0)

trixi2vtk(shape_sampled)

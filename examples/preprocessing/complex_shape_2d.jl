using TrixiParticles

particle_spacing = 0.05

file = "hexagon"
filename = joinpath("examples", "preprocessing", file * ".asc")

# Returns `Shape`
shape = load_shape(filename)

# Returns `InitialCondition`.
shape_sampled = ComplexShape(shape; particle_spacing, density=1.0)

trixi2vtk(shape)
trixi2vtk(shape_sampled)

using TrixiParticles

particle_spacing = 0.5

dir = joinpath("Data", "stl-files", "examples")
filename = joinpath(expanduser("~/") * dir, "cube.stl")

# Returns `Shape`
shape = load_shape(filename)

# Returns `InitialCondition`.
shape_sampled = ComplexShape(shape; particle_spacing, density=1.0)

trixi2vtk(shape_sampled.coordinates, filename="coords");

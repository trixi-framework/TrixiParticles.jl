using TrixiParticles

particle_spacing = 0.05

file = "hexagon"
filename = joinpath("examples", "preprocessing", file * ".asc")

# Returns `Shape`
geomtery = load_geomtery(filename)

# Returns `InitialCondition`.
shape_sampled = ComplexShape(geometry; particle_spacing, density=1.0)

trixi2vtk(shape_sampled)

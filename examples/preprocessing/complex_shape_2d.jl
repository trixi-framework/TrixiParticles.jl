using TrixiParticles

particle_spacing = 0.05

file = "hexagon"
filename = joinpath("examples", "preprocessing", file * ".asc")

geometry = load_geometry(filename)

trixi2vtk(geometry)

# Returns `InitialCondition`.
shape_sampled = ComplexShape(geometry; particle_spacing, density=1.0)

trixi2vtk(shape_sampled)

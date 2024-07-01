using TrixiParticles

particle_spacing = 0.05

file = "hexagon"
filename = joinpath("examples", "preprocessing", file * ".asc")

# Returns `Shape`
shape = load_shape(filename)

# Returns `InitialCondition`.
shape_sampled = ComplexShape(shape; sample_boundary=false, particle_spacing, density=1.0,
                             boundary_thickness=5particle_spacing)

trixi2vtk(shape)
trixi2vtk(shape_sampled.initial_condition; filename="initial_condition_fluid")
trixi2vtk(shape_sampled.signed_distance_field)

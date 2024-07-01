using TrixiParticles

particle_spacing = 0.05

file = "sphere"
filename = joinpath("examples", "preprocessing", file * ".stl")

boundary_thickness = 5particle_spacing

# Returns `Shape`
shape = load_shape(filename)
# Returns `InitialCondition`.
shape_sampled = ComplexShape(shape; particle_spacing, density=1.0,
                             sample_boundary=false,
                             point_in_shape_algorithm=WindingNumberJacobson(; shape,
                                                                            # winding_number_factor=0.4,
                                                                            hierarchical_winding=true))

trixi2vtk(shape_sampled.initial_condition, filename="initial_condition_fluid")

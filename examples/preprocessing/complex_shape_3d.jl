using TrixiParticles

particle_spacing = 0.05

filename = joinpath("examples", "preprocessing", "sphere.stl")

# The following triangle mesh is corrupt.
# For more robustness, use `winding_number_factor=0.4`.
# filename = joinpath("examples", "preprocessing", "drive_gear.stl")

# Returns `Shape`
shape = load_shape(filename)

# Returns `InitialCondition`.
shape_sampled = ComplexShape(shape; particle_spacing, density=1.0,
                             point_in_shape_algorithm=WindingNumberJacobson(; shape,
                                                                            hierarchical_winding=true))

trixi2vtk(shape_sampled)

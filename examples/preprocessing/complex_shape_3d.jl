using TrixiParticles

particle_spacing = 0.05

file = "sphere"
filename = joinpath("examples", "preprocessing", file * ".stl")

# The following triangle mesh is corrupt.
# For more robustness, use `winding_number_factor=0.4`.
# filename = joinpath("examples", "preprocessing", "drive_gear.stl")

# Returns `Shape`
geometry = load_geometry(filename)

point_in_geometry_algorithm = WindingNumberJacobson(; geometry,
                                                    # winding_number_factor=0.4,
                                                    hierarchical_winding=true)

# Returns `InitialCondition`.
shape_sampled = ComplexShape(geometry; particle_spacing, density=1.0,
                             point_in_geometry_algorithm)

trixi2vtk(shape_sampled)

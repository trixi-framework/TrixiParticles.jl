using TrixiParticles

particle_spacing = 0.5

filename = joinpath(examples_dir(), "preprocessing", "fluid.asc")

data = TrixiParticles.read_in_2d(; filename)

# Returns `InitialCondition`.
test = ComplexShape(; filename, particle_spacing, density=1.0,
                    point_in_poly_algorithm=WindingNumberJacobson())

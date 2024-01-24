using TrixiParticles

particle_spacing = 0.1

dir = joinpath("Data", "stl-files", "examples")
filename = joinpath(expanduser("~/") * dir, "bar.stl")

# Returns `InitialCondition`.
test = ComplexShape(; filename, particle_spacing, density=1.0,
                    point_in_poly_algorithm=WindingNumberJacobson())

trixi2vtk(test.coordinates, filename="coords")

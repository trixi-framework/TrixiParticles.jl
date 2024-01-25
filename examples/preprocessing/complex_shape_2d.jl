using TrixiParticles

particle_spacing = 2.0

filename = joinpath(examples_dir(), "preprocessing", "inverted_curve_open.asc")

data = TrixiParticles.read_in_2d(; filename)

# Returns `InitialCondition`.
#= test = ComplexShape(; filename, particle_spacing, density=1.0,
                    point_in_shape_algorithm=WindingNumberHorman()) =#

point_in_shape_algorithm = WindingNumberJacobson(; winding_number_factor=0.5)
points = TrixiParticles.read_in_2d(; filename)
shape = TrixiParticles.Polygon(points)
grid = TrixiParticles.particle_grid(shape.vertices, particle_spacing)

inpoly, winding_numbers = point_in_shape_algorithm(shape, grid)

coordinates = grid[:, inpoly]

trixi2vtk(grid, filename="winding_number", winding_number=winding_numbers)
trixi2vtk(coordinates, filename="coords")
trixi2vtk(points, filename="points")

using TrixiParticles
using DelimitedFiles

#polygon = TrixiParticles.Polygon([1.8 2.4 6.2 8.5 5.1 3.4 1.8;
#                                   4.2 1.7 0.8 3.2 5.1 0.2 4.2])

# Scale the input shape
factor = 1.0 # 1.0e-2

particle_spacing = 0.1

# Read in the ASCII file as an Tuple containing the coordinates of the points and the
# header.
# `readdlm(...)[1][:, 1:2]` access only the point coordinates and only x and y direction.
points_test = readdlm(joinpath(examples_dir(), "preprocessing", "test.asc"), ' ',
                      Float64, '\n', header=true)[1][:, 1:2] .* factor

shape_test = TrixiParticles.Polygon(points_test')

# Returns `InitialCondition` (see docs).
test = TrixiParticles.sample(; shape=shape_test, particle_spacing,
                             density=1.0,
                             point_in_poly=TrixiParticles.WindingNumberJacobson())

trixi2vtk(points_test', filename="points")
trixi2vtk(test.coordinates, filename="coords")

using TrixiParticles
using FileIO

particle_spacing = 0.1

dir = joinpath("Data", "stl-files", "examples")
mesh = load(joinpath(expanduser("~/") * dir, "bar.stl"))

triangle_mesh = TrixiParticles.TriangleMesh(mesh)

initial_condition = TrixiParticles.sample(; shape=triangle_mesh, particle_spacing,
                                          density=1.0,
                                          point_in_poly=TrixiParticles.WindingNumberJacobson())

# trixi2vtk(coords, filename="test_coords")

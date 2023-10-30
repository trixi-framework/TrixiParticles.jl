using TrixiParticles
using FileIO

dir = joinpath("Data", "stl-files", "examples")
mesh = load(joinpath(expanduser("~/") * dir, "bar.stl"))

polygon = TrixiParticles.Polygon([1.8 2.4 6.2 8.5 5.1 3.4 1.8;
                                  4.2 1.7 0.8 3.2 5.1 0.2 4.2])

triangle_mesh = TrixiParticles.TriangleMesh(mesh)

coords = TrixiParticles.sample(triangle_mesh,
                               point_in_poly=TrixiParticles.WindingNumberJacobson(),
                               1.0)

#coords = TrixiParticles.sample(polygon, 0.1, point_in_poly=TrixiParticles.WindingNumberJacobson())
trixi2vtk(coords, filename="test_coords")
trixi2vtk(polygon.vertices, filename="polygon_coords")

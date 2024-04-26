using TrixiParticles
using OrdinaryDiffEq

tspan = (0.0, 10.0)

particle_spacing = 0.5

filename = joinpath("out_preprocessing", "hexagon.asc")

# Returns `Shape`
shape = load_shape(filename)

# Returns `InitialCondition`.
shape_sampled = ComplexShape(shape; particle_spacing, density=1.0)

shape_packed = ParticlePacking(shape_sampled, shape;
                               tlsph=true, maxiters=100,
                               neighborhood_search=true,
                               background_pressure=1000,
                               precalculate_sdf=true,
                               solution_saving_callback=SolutionSavingCallback(interval=1,
                                                                               output_directory="out"),
                               info_callback=InfoCallback(interval=50))
#point_in_shape_algorithm=WindingNumberHorman())

trixi2vtk(shape_sampled.coordinates, filename="coords")
points = TrixiParticles.read_in_2d(; filename)
trixi2vtk(points, filename="points");

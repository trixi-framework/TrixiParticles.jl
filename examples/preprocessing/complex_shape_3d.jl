using TrixiParticles
#using OrdinaryDiffEq

tspan = (0.0, 50.0)
particle_spacing = 0.02

dir = joinpath("Data", "stl-files", "examples")
filename = joinpath(expanduser("~/") * dir, "aorta.stl")

# Returns `Shape`
shape = load_shape(filename)

# Returns `InitialCondition`.
shape_sampled = ComplexShape(shape; particle_spacing, density=1.0)

shape_packed = ParticlePacking(shape_sampled, shape;
                               tlsph=true, maxiters=100,
                               neighborhood_search=true,
                               background_pressure=100,
                               precalculate_sdf=true,
                               solution_saving_callback=SolutionSavingCallback(dt=0.02,
                                                                               output_directory="out"),
                               info_callback=InfoCallback(interval=50))
trixi2vtk(shape_sampled.coordinates, filename="coords")

using TrixiParticles

particle_spacing = 0.05

filename = "sphere"
file = joinpath("examples", "preprocessing", "data", filename * ".stl")

geometry = load_geometry(file; parallelization_backend=false,
                         element_type=typeof(particle_spacing))

trixi2vtk(geometry)

# Returns `InitialCondition`
shape_sampled = ComplexShape(geometry; particle_spacing, density=1.0,
                             point_in_geometry_algorithm=WindingNumberJacobson(geometry))

trixi2vtk(shape_sampled)

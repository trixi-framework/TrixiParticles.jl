using TrixiParticles

particle_spacing = 0.05

filename = "cube"
file = joinpath("examples", "preprocessing", "data", filename * ".stl")

geometry = load_geometry(file)


filename_diff = "sphere"
file_diff = joinpath("examples", "preprocessing", "data", filename_diff * ".stl")

geometry_diff = load_geometry(file_diff)

point_in_geometry_algorithm = WindingNumberJacobson(; geometry,
                                                    # winding_number_factor=0.4,
                                                    hierarchical_winding=true)

# Returns `InitialCondition`
shape_sampled = ComplexShape(geometry; particle_spacing, density=1.0,
                             point_in_geometry_algorithm, exclude_geometry=geometry_diff)

trixi2vtk(shape_sampled)

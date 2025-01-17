using TrixiParticles

particle_spacing = 0.05

filename = "sphere"
file = joinpath("examples", "preprocessing", "data", filename * ".stl")

geometry = load_geometry(file)

point_in_geometry_algorithm = WindingNumberJacobson(; geometry,
                                                    # winding_number_factor=0.4,
                                                    hierarchical_winding=true)

# Returns `InitialCondition`
shape_sampled = ComplexShape(geometry; particle_spacing, density=1.0,
                             boundary_thickness=5 * particle_spacing,
                             create_signed_distance_field=true,
                             sample_boundary=false, point_in_geometry_algorithm)

trixi2vtk(shape_sampled.initial_condition, filename="initial_condition_fluid")
trixi2vtk(shape_sampled.signed_distance_field)

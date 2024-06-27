using TrixiParticles

particle_spacing = 0.05

file = "sphere"
filename = joinpath("examples", "preprocessing", file * ".stl")

sample_boundary = false

boundary_thickness = 5particle_spacing

# Returns `Shape`
shape = load_shape(filename)

signed_distance_field = SignedDistanceField(shape, particle_spacing;
                                            max_signed_distance=boundary_thickness,
                                            use_for_boundary_packing=sample_boundary,
                                            neighborhood_search=true)
trixi2vtk(signed_distance_field)

# Returns `InitialCondition`.
shape_sampled = ComplexShape(shape; particle_spacing, density=1.0,
                             signed_distance_field, sample_boundary,
                             point_in_shape_algorithm=WindingNumberJacobson(; shape,
                                                                            # winding_number_factor=0.4,
                                                                            hierarchical_winding=true))

if sample_boundary
    trixi2vtk(shape_sampled.fluid, filename="initial_condition_fluid")
    trixi2vtk(shape_sampled.boundary, filename="initial_condition_boundary")
else
    trixi2vtk(shape_sampled)
end

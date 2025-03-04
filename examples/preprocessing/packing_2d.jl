using TrixiParticles
using OrdinaryDiffEq

filename = "circle"
file = pkgdir(TrixiParticles, "examples", "preprocessing", "data", filename * ".asc")

# ==========================================================================================
# ==== Packing parameters
save_intervals = false
tlsph = true

# ==========================================================================================
# ==== Resolution
particle_spacing = 0.03

# The following depends on the sampling of the particles. In this case `boundary_thickness`
# means literally the thickness of the boundary packed with boundary particles and *not*
# how many rows of boundary particles will be sampled.
boundary_thickness = 8particle_spacing

# ==========================================================================================
# ==== Load complex geometry
density = 1000.0

geometry = load_geometry(file; element_type=typeof(particle_spacing))

signed_distance_field = SignedDistanceField(geometry, particle_spacing;
                                            use_for_boundary_packing=true,
                                            max_signed_distance=boundary_thickness)

point_in_geometry_algorithm = WindingNumberJacobson(; geometry,
                                                    winding_number_factor=0.4,
                                                    hierarchical_winding=true)
# Returns `InitialCondition`
shape_sampled = ComplexShape(geometry; particle_spacing, density,
                             point_in_geometry_algorithm)

# Returns `InitialCondition`
boundary_sampled = sample_boundary(signed_distance_field; boundary_density=density,
                                   boundary_thickness, tlsph)

trixi2vtk(shape_sampled)
trixi2vtk(boundary_sampled, filename="boundary")

# ==========================================================================================
# ==== Packing

# Large `background_pressure` can cause high accelerations. That is, the adaptive
# time-stepsize will be adjusted properly. We found that the following order of
# `background_pressure` result in appropriate stepsizes.
background_pressure = 1e6 * particle_spacing^ndims(geometry)

min_corner = minimum(stack(signed_distance_field.positions), dims=2)
max_corner = maximum(stack(signed_distance_field.positions), dims=2)
cell_list = FullGridCellList(; min_corner, max_corner)
neighborhood_search = GridNeighborhoodSearch{2}(; cell_list)

packing_system = ParticlePackingSystem(shape_sampled;
                                       signed_distance_field, tlsph=tlsph,
                                       background_pressure, neighborhood_search)

boundary_system = ParticlePackingSystem(boundary_sampled;
                                        is_boundary=true, signed_distance_field,
                                        tlsph=tlsph, boundary_compress_factor=0.8,
                                        background_pressure, neighborhood_search)

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(packing_system, boundary_system; neighborhood_search)

# Use a high `tspan` to guarantee that the simulation runs at least for `maxiters`
tspan = (0, 10.0)
ode = semidiscretize(semi, tspan, data_type=nothing)

# Use this callback to stop the simulation when it is sufficiently close to a steady state
# steady_state = SteadyStateReachedCallback(; interval=1, interval_size=10,
#                                           abstol=1.0e-5, reltol=1.0e-3)
steady_state = nothing

info_callback = InfoCallback(interval=50)

saving_callback = save_intervals ?
                  SolutionSavingCallback(interval=10, prefix="", ekin=kinetic_energy) :
                  nothing

callbacks = CallbackSet(UpdateCallback(), saving_callback, info_callback, steady_state)

sol = solve(ode, RDPK3SpFSAL35();
            save_everystep=false, maxiters=1000, callback=callbacks, dtmax=1e-2)

packed_ic = InitialCondition(sol, packing_system, semi)
packed_boundary_ic = InitialCondition(sol, boundary_system, semi)

trixi2vtk(packed_ic, filename="initial_condition_packed")
trixi2vtk(packed_boundary_ic, filename="initial_condition_boundary_packed")

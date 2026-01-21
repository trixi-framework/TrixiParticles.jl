# ==========================================================================================
# 2D Particle Packing within a Complex Geometry
#
# This example demonstrates how to:
# 1. Load a 2D geometry (e.g., a circle defined by a boundary curve).
# 2. Generate an initial, potentially overlapping, distribution of "fluid" particles
#    inside the geometry and "boundary" particles forming a layer around it.
# 3. Use the `ParticlePackingSystem` to run a pseudo-SPH simulation that relaxes
#    the particle positions, achieving a more uniform and non-overlapping distribution.
# 4. Visualize the initial and packed particle configurations.
#
# This is a common preprocessing step to create stable initial conditions for SPH simulations.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq, Plots

filename = "circle"
file = pkgdir(TrixiParticles, "examples", "preprocessing", "data", filename * ".asc")

# ==========================================================================================
# ==== Packing parameters
place_on_shell = false

# ==========================================================================================
# ==== Resolution
particle_spacing = 0.1

# The following depends on the sampling of the particles. In this case `boundary_thickness`
# means literally the thickness of the boundary packed with boundary particles and *not*
# how many rows of boundary particles will be sampled.
boundary_thickness = 5 * particle_spacing

# ==========================================================================================
# ==== Load complex geometry
density = 1.0

geometry = load_geometry(file)

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
                                   boundary_thickness, place_on_shell=place_on_shell)

trixi2vtk(shape_sampled)
trixi2vtk(boundary_sampled, filename="boundary")

# ==========================================================================================
# ==== Packing

# A larger `background_pressure` makes the packing happen faster in physical time,
# which results in a correspondingly smaller time step.
# Essentially, the `background_pressure` just scales the physical time,
# and can therefore arbitrarily be set to 1.
background_pressure = 1.0

smoothing_length = 0.8 * particle_spacing
packing_system = ParticlePackingSystem(shape_sampled; smoothing_length=smoothing_length,
                                       signed_distance_field, place_on_shell=place_on_shell,
                                       background_pressure)

boundary_system = ParticlePackingSystem(boundary_sampled; smoothing_length=smoothing_length,
                                        is_boundary=true, signed_distance_field,
                                        place_on_shell=place_on_shell,
                                        boundary_compress_factor=0.8,
                                        background_pressure)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(packing_system, boundary_system)

# Use a high `tspan` to guarantee that the simulation runs at least for `maxiters`
tspan = (0, 10.0)
ode = semidiscretize(semi, tspan)

# Use this callback to stop the simulation when it is sufficiently close to a steady state
steady_state = SteadyStateReachedCallback(; interval=10, interval_size=200,
                                          abstol=1.0e-7, reltol=1.0e-6)

info_callback = InfoCallback(interval=50)

save_intervals = false
saving_callback = save_intervals ?
                  SolutionSavingCallback(interval=10, prefix="", ekin=kinetic_energy) :
                  nothing

pp_cb_ekin = PostprocessCallback(; ekin=kinetic_energy, interval=1,
                                 filename="kinetic_energy", write_file_interval=50)

callbacks = CallbackSet(UpdateCallback(), saving_callback, info_callback, steady_state,
                        pp_cb_ekin)

sol = solve(ode, RDPK3SpFSAL35();
            save_everystep=false, maxiters=1000, callback=callbacks, dtmax=1e-2)

packed_ic = InitialCondition(sol, packing_system, semi)
packed_boundary_ic = InitialCondition(sol, boundary_system, semi)

trixi2vtk(packed_ic, filename="initial_condition_packed")
trixi2vtk(packed_boundary_ic, filename="initial_condition_boundary_packed")

shape = Plots.Shape(stack(geometry.vertices)[1, :], stack(geometry.vertices)[2, :])
p1 = plot(shape_sampled, markerstrokewidth=1, label=nothing, layout=(1, 2))
plot!(p1, shape, color=nothing, label=nothing, linewidth=2, subplot=1)
plot!(p1, packed_ic, markerstrokewidth=1, label=nothing, subplot=2)
plot!(p1, shape, color=nothing, label=nothing, linewidth=2, subplot=2, size=(850, 400))

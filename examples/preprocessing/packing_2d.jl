using TrixiParticles
using OrdinaryDiffEq

filename = "circle"
file = joinpath("examples", "preprocessing", "data", filename * ".asc")

# ==========================================================================================
# ==== Packing parameters
maxiters = 200
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
# ==== Load complex shape
geometry = load_geometry(file)

point_in_geometry_algorithm = WindingNumberJacobson(; geometry,
                                                    winding_number_factor=0.4,
                                                    hierarchical_winding=true)
# Returns `InitialCondition`
shape_sampled = ComplexShape(geometry; particle_spacing, density=1000.0,
                             store_winding_number=true, sample_boundary=true,
                             boundary_thickness, tlsph=tlsph,
                             create_signed_distance_field=true,
                             grid_offset=0.25 * particle_spacing,
                             point_in_geometry_algorithm)

trixi2vtk(shape_sampled.initial_condition)
trixi2vtk(shape_sampled.initial_condition_boundary, filename="initial_condition_boundary")

# ==========================================================================================
# ==== Packing

# Large `background_pressure` can cause high accelerations. That is, the adaptive
# time-stepsize will be adjusted properly. We found that the following order of
# `background_pressure` result in appropriate time-stepsizes.
background_pressure = 1e6 * particle_spacing^2

packing_system = ParticlePackingSystem(shape_sampled; tlsph=tlsph,
                                       background_pressure)

boundary_system = ParticlePackingSystem(shape_sampled; tlsph=tlsph,
                                        is_boundary=true,
                                        boundary_compress_factor=0.8,
                                        background_pressure)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(packing_system, boundary_system)

# Use a high `tspan` to guarantee that the simulation runs at least for `maxiters`
tspan = (0, 10.0)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
saving_callback = save_intervals ? SolutionSavingCallback(interval=10, prefix="") : nothing

callbacks = CallbackSet(UpdateCallback(), saving_callback, info_callback)

sol = solve(ode, RK4();
            save_everystep=false, maxiters=maxiters, callback=callbacks, dtmax=1e-2)

packed_ic = InitialCondition(sol, packing_system, semi)
packed_boundary_ic = InitialCondition(sol, boundary_system, semi)

v_ode, u_ode = sol.u[end].x
u = TrixiParticles.wrap_u(u_ode, packing_system, semi)
TrixiParticles.summation_density!(packing_system, semi, u, u_ode, packed_ic.density)

u = TrixiParticles.wrap_u(u_ode, boundary_system, semi)
TrixiParticles.summation_density!(boundary_system, semi, u, u_ode,
                                  packed_boundary_ic.density)

trixi2vtk(packed_ic, filename="initial_condition_packed")
trixi2vtk(packed_boundary_ic, filename="initial_condition_boundary_packed")

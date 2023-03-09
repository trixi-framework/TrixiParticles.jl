# 2D dam break simulation based on
#
# S. Marrone, M. Antuono, A. Colagrossi, G. Colicchio, D. le Touzé, G. Graziani.
# "δ-SPH model for simulating violent impact flows".
# In: Computer Methods in Applied Mechanics and Engineering, Volume 200, Issues 13–16 (2011), pages 1526–1542.
# https://doi.org/10.1016/J.CMA.2010.12.016

using Pixie
using OrdinaryDiffEq

# ==========================================================================================
# ==== Reference Values

gravity = 9.81
atmospheric_pressure = 100000.0
incompressible_gamma = 7
ambient_temperature = 293.15

# ==========================================================================================
# ==== Fluid

water_width = 2.0
water_height = 1.0
water_density = 1000.0

sound_speed = 20 * sqrt(gravity * water_height)

state_equation = StateEquationCole(sound_speed, incompressible_gamma, water_density, atmospheric_pressure,
                                   background_pressure=atmospheric_pressure)

viscosity = ArtificialViscosityMonaghan(0.02, 0.0)
water_at_rest = State(water_density, atmospheric_pressure, ambient_temperature)

# ==========================================================================================
# ==== Particle Setup

particle_spacing = 0.05

# Spacing ratio between fluid and boundary particles
beta = 1
boundary_layers = 4

smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

tank_width = floor(5.366 / particle_spacing * beta) * particle_spacing / beta
tank_height = 6

setup = RectangularTank(particle_spacing, (water_width, water_height),
                        (tank_width, tank_height), water_density,
                        n_layers=boundary_layers, spacing_ratio=beta)

# Move right boundary
# Recompute the new water column width since the width has been rounded in `RectangularTank`.
new_wall_position = (setup.n_particles_per_dimension[1] + 1) * particle_spacing
reset_faces = (false, true, false, false)
positions = (0, new_wall_position, 0, 0)

reset_wall!(setup, reset_faces, positions)

# ==========================================================================================
# ==== Boundary models

boundary_model = BoundaryModelDummyParticles(setup.boundary_densities,
                                             setup.boundary_masses, state_equation,
                                             AdamiPressureExtrapolation(), smoothing_kernel,
                                             smoothing_length)

# K = 9.81 * water_height
# boundary_model = BoundaryModelMonaghanKajtar(K, beta, particle_spacing / beta,
#                                              setup.boundary_masses)

# ==========================================================================================
# ==== Containers

particle_container = FluidParticleContainer(setup,
                                            SummationDensity(), state_equation,
                                            smoothing_kernel, smoothing_length, water_at_rest,
                                            viscosity=viscosity,
                                            acceleration=(0.0, -gravity),
                                            surface_tension=SurfaceTensionAkinci(surface_tension_coefficient=0.0005),
                                            store_options=StoreAll())

boundary_container = BoundaryParticleContainer(setup.boundary_coordinates, boundary_model)

# ==========================================================================================
# ==== Simulation

# 1. Initialization

semi = Semidiscretization(particle_container, boundary_container,
                          neighborhood_search=SpatialHashingSearch,
                          damping_coefficient=1e-5)

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=100)

# activate to save
#saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:0.01:5.0,
#                                                       index=(v, u, t, container) -> Pixie.eachparticle(container))
#callbacks = CallbackSet(summary_callback, alive_callback, saving_callback)

callbacks = CallbackSet(summary_callback, alive_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Enable threading of the RK method for better performance on multiple threads.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so with Monaghan-Kajtar BC because forces
# become extremely large when fluid particles are very close to boundary particles,
# and the time integration method interprets this as an instability.
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-5, # Default abstol is 1e-6 (may needs to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may needs to be tuned to prevent boundary penetration)
            dtmax=1e-3, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()

#pixie2vtk(saved_values)

# 2. Main Simulation

# Move right boundary
positions = (0, tank_width, 0, 0)
reset_wall!(setup, reset_faces, positions)

# Run full simulation
tspan = (0.0, 5.7 / sqrt(gravity))

# Use solution of the relaxing step as initial coordinates
restart_with!(semi, sol)

semi = Semidiscretization(particle_container, boundary_container,
                          neighborhood_search=SpatialHashingSearch)
ode = semidiscretize(semi, tspan)

saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:0.02:1000.0,
                                                       index=(v, u, t, container) -> Pixie.eachparticle(container))

callbacks = CallbackSet(summary_callback, alive_callback, saving_callback)

# See above for an explanation of the parameter choice
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-6, # Default abstol is 1e-6 (may needs to be tuned to prevent boundary penetration)
            reltol=1e-5, # Default reltol is 1e-3 (may needs to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()

# activate to save to vtk
pixie2vtk(saved_values)

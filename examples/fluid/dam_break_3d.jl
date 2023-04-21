using TrixiParticles
using OrdinaryDiffEq

gravity = -9.81

# ==========================================================================================
# ==== Fluid

particle_spacing = 0.08

# Spacing ratio between fluid and boundary particles
beta = 1
boundary_layers = 3

water_width = floor(2.0 / particle_spacing) * particle_spacing # x-direction
water_height = floor(1.0 / particle_spacing) * particle_spacing # y-direction
water_length = floor(1.0 / particle_spacing) * particle_spacing # z-direction
water_density = 1000.0

tank_width = floor(5.366 / particle_spacing * beta) * particle_spacing / beta
tank_height = floor(4.0 / particle_spacing) * particle_spacing
tank_length = floor(1.0 / particle_spacing) * particle_spacing

sound_speed = 20 * sqrt(9.81 * water_height)

smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{3}()

state_equation = StateEquationCole(sound_speed, 7, water_density, 100000.0,
                                   background_pressure=100000.0)
scheme = WCSPH(state_equation)

viscosity = ArtificialViscosityMonaghan(0.02, 0.0)

setup = RectangularTank(particle_spacing, (water_width, water_height, water_length),
                        (tank_width, tank_height, tank_length), water_density,
                        n_layers=boundary_layers, spacing_ratio=beta)

# Move right boundary
new_wall_position = (setup.n_particles_per_dimension[1] + 1) * particle_spacing
reset_faces = (false, true, false, false, false, false)
positions = (0, new_wall_position, 0, 0, 0, 0)

reset_wall!(setup, reset_faces, positions)

# ==========================================================================================
# ==== Boundary models

boundary_model = BoundaryModelDummyParticles(scheme, setup.boundary_densities,
                                             setup.boundary_masses,
                                             state_equation=state_equation,
                                             AdamiPressureExtrapolation(), smoothing_kernel,
                                             smoothing_length)
# K = 9.81 * water_height
# boundary_model = BoundaryModelMonaghanKajtar(K, beta, particle_spacing / beta,
#                                              setup.boundary_masses)

# ==========================================================================================
# ==== Containers

particle_container = FluidParticleContainer(scheme, setup, ContinuityDensity(),
                                            smoothing_kernel, smoothing_length,
                                            viscosity=viscosity,
                                            acceleration=(0.0, gravity, 0.0))

boundary_container = BoundaryParticleContainer(setup.boundary_coordinates, boundary_model)

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(particle_container, boundary_container,
                          neighborhood_search=SpatialHashingSearch,
                          damping_coefficient=1e-5)

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=10)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Enable threading of the RK method for better performance on multiple threads.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so with Monaghan-Kajtar BC because forces
# become extremely large when fluid particles are very close to boundary particles,
# and the time integration method interprets this as an instability.
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-3, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=info_callback);

# Move right boundary
positions = (0, tank_width, 0, 0)
reset_wall!(setup, reset_faces, positions)

# Run full simulation
tspan = (0.0, 5.7 / sqrt(9.81))

# Use solution of the relaxing step as initial coordinates
restart_with!(semi, sol)

semi = Semidiscretization(particle_container, boundary_container,
                          neighborhood_search=SpatialHashingSearch)
ode = semidiscretize(semi, tspan)

saving_callback = SolutionSavingCallback(dt=0.02)
callbacks = CallbackSet(info_callback, saving_callback)

# See above for an explanation of the parameter choice
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);

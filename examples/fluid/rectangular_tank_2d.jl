using TrixiParticles
using OrdinaryDiffEq

gravity = -9.81

# ==========================================================================================
# ==== Fluid

particle_spacing = 0.02

# Ratio of fluid particle spacing to boundary particle spacing
beta = 1
boundary_layers = 3

water_width = 2.0
water_height = 0.9
water_density = 1000.0

tank_width = 2.0
tank_height = 1.0

sound_speed = 10 * sqrt(9.81 * water_height)
state_equation = StateEquationCole(sound_speed, 7, water_density, 100000.0,
                                   background_pressure=100000.0)

smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

viscosity = ArtificialViscosityMonaghan(0.02, 0.0)

setup = RectangularTank(particle_spacing, (water_width, water_height),
                        (tank_width, tank_height), water_density,
                        n_layers=boundary_layers, spacing_ratio=beta)

# ==========================================================================================
# ==== Boundary models

boundary_model = BoundaryModelDummyParticles(setup.boundary_densities,
                                             setup.boundary_masses, state_equation,
                                             AdamiPressureExtrapolation(),
                                             smoothing_kernel,
                                             smoothing_length)

# K = 9.81 * water_height
# boundary_model = BoundaryModelMonaghanKajtar(K, beta, particle_spacing / beta,
#                                              setup.boundary_masses)

# ==========================================================================================
# ==== Containers

particle_container = FluidParticleContainer(setup, ContinuityDensity(), state_equation,
                                            smoothing_kernel, smoothing_length,
                                            water_density,
                                            viscosity=viscosity,
                                            acceleration=(0.0, gravity))

boundary_container = BoundaryParticleContainer(setup.boundary_coordinates, boundary_model)

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(particle_container, boundary_container,
                          neighborhood_search=SpatialHashingSearch,
                          damping_coefficient=1e-5)

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=10)
saving_callback = SolutionSavingCallback(dt=0.02)

callbacks = CallbackSet(info_callback, saving_callback)

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
            save_everystep=false, callback=callbacks);

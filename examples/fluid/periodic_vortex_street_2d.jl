using TrixiParticles
using OrdinaryDiffEq

gravity = -9.81

# ==========================================================================================
# ==== Fluid

particle_spacing = 0.005

# Ratio of fluid particle spacing to boundary particle spacing
beta = 1
boundary_layers = 3

water_density = 1000.0

smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

tank_width = 0.96
tank_height = 0.48

water_width = tank_width
water_height = tank_height

sound_speed = 10.0
state_equation = StateEquationCole(sound_speed, 7, water_density, 100_000.0,
                                   background_pressure=100_000.0)

viscosity = ArtificialViscosityMonaghan(0.02, 0.0)

tank = RectangularTank(particle_spacing, (water_width, water_height),
                       (tank_width, tank_height), water_density,
                       n_layers=boundary_layers, spacing_ratio=beta,
                       faces=(false, false, true, true), init_velocity=(1.0, 0.0))

hollow_sphere = SphereShape(particle_spacing, 0.06, (0.2, 0.24), water_density,
                            n_layers=3, sphere_type=RoundSphere())

filled_sphere = SphereShape(particle_spacing, 0.06, (0.2, 0.24), water_density,
                            sphere_type=RoundSphere())

boundary = union(tank.boundary, hollow_sphere)

fluid = setdiff(tank.fluid, filled_sphere)

# ==========================================================================================
# ==== Boundary models

boundary_model = BoundaryModelDummyParticles(boundary.density,
                                             boundary.mass, state_equation,
                                             AdamiPressureExtrapolation(),
                                             smoothing_kernel,
                                             smoothing_length)

# K = 9.81 * water_height
# boundary_model = BoundaryModelMonaghanKajtar(K, beta, particle_spacing / beta,
#                                              boundary.mass)

# ==========================================================================================
# ==== Systems

fluid_system = WeaklyCompressibleSPHSystem(fluid, SummationDensity(), state_equation,
                                           smoothing_kernel, smoothing_length,
                                           viscosity=viscosity)

boundary_system = BoundarySPHSystem(boundary, boundary_model)

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(fluid_system, boundary_system,
                          neighborhood_search=SpatialHashingSearch,
                          periodic_box_min_corner=[0.0, -0.24],
                          periodic_box_max_corner=[0.96, 0.72])

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
            abstol=1e-8, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);

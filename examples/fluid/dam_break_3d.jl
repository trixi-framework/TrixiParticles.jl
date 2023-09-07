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

viscosity = ArtificialViscosityMonaghan(0.02, 0.0)

tank = RectangularTank(particle_spacing, (water_width, water_height, water_length),
                       (tank_width, tank_height, tank_length), water_density,
                       n_layers=boundary_layers, spacing_ratio=beta)

# Move +x boundary.
# Use the new fluid size, since it might have been rounded in `RectangularTank`.
reset_faces = (false, true, false, false, false, false)
positions = (0, tank.fluid_size[1], 0, 0, 0, 0)

reset_wall!(tank, reset_faces, positions)

# ==========================================================================================
# ==== Boundary models

boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             AdamiPressureExtrapolation(), smoothing_kernel,
                                             smoothing_length)
# K = 9.81 * water_height
# boundary_model = BoundaryModelMonaghanKajtar(K, beta, particle_spacing / beta,
#                                              tank.boundary.mass)

# ==========================================================================================
# ==== Systems

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, ContinuityDensity(), state_equation,
                                           smoothing_kernel, smoothing_length,
                                           viscosity=viscosity,
                                           acceleration=(0.0, gravity, 0.0))

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(fluid_system, boundary_system,
                          neighborhood_search=GridNeighborhoodSearch,
                          damping_coefficient=1e-5)

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=10)
saving_callback_relaxation = SolutionSavingCallback(dt=0.02, prefix="relaxation")
callbacks_relaxation = CallbackSet(info_callback, saving_callback_relaxation)

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
            save_everystep=false, callback=callbacks_relaxation);

# Move right boundary
positions = (0, tank.tank_size[1], 0, 0)
reset_wall!(tank, reset_faces, positions)

# Run full simulation
tspan = (0.0, 5.7 / sqrt(9.81))

# Use solution of the relaxing step as initial coordinates
restart_with!(semi, sol)

semi = Semidiscretization(fluid_system, boundary_system,
                          neighborhood_search=GridNeighborhoodSearch)
ode = semidiscretize(semi, tspan)

saving_callback = SolutionSavingCallback(dt=0.02)
callbacks = CallbackSet(info_callback, saving_callback)

# See above for an explanation of the parameter choice
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);

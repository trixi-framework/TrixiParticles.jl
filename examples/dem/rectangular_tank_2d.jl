using TrixiParticles
using OrdinaryDiffEq

gravity = -9.81

# ==========================================================================================
# ==== Fluid

particle_spacing = 0.1

# Ratio of fluid particle spacing to boundary particle spacing
beta = 1
boundary_layers = 1

water_width = 2.0
water_height = 2.0
water_density = 1000.0

tank_width = 2.0
tank_height = 4.0

tank = RectangularTank(particle_spacing, (water_width, water_height),
                       (tank_width, tank_height), water_density,
                       n_layers=boundary_layers, spacing_ratio=beta)

# ==========================================================================================
# ==== Boundary models

# boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
#                                              state_equation=state_equation,
#                                              AdamiPressureExtrapolation(),
#                                              smoothing_kernel, smoothing_length)

# ==========================================================================================
# ==== Systems

solid_system = DEMSystem(tank.fluid, 0.1, acceleration=(0.0, gravity))

# boundary_system = BoundaryDEMSystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(solid_system, neighborhood_search=GridNeighborhoodSearch)

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
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

using TrixiParticles
using OrdinaryDiffEq

gravity = 0.0
reynolds_number = 100.0
particle_spacing = 0.02

# ==========================================================================================
# ==== Fluid

water_width = 1.0
water_height = 1.0
water_density = 1.0

tank_width = 1.0
tank_height = 1.0

velocity_lid = 1.0
sound_speed = 10 * velocity_lid

pressure = sound_speed^2 * water_density

smoothing_length = 1.0 * particle_spacing
smoothing_kernel = SchoenbergQuinticSplineKernel{2}()

nu = velocity_lid / reynolds_number
viscosity = ViscosityAdami(nu)

tank = RectangularTank(particle_spacing, (water_width, water_height),
                       (tank_width, tank_height), water_density,
                       n_layers=3, spacing_ratio=1, faces=(true, true, true, false),
                       pressure=pressure)

# ==========================================================================================
# ==== Lid

lid_position = 0.0 - particle_spacing * 4
lid_length = tank.n_particles_per_dimension[1] + 8

lid = RectangularShape(particle_spacing, (lid_length, 3),
                       (lid_position, water_height), water_density)

f_y(t) = 0.0
f_x(t) = velocity_lid * t

is_moving(t) = true

movement = BoundaryMovement((f_x, f_y), is_moving)

# ==========================================================================================
# ==== Boundary models

boundary_model_tank = BoundaryModelDummyParticles(tank.boundary.density,
                                                  tank.boundary.mass,
                                                  AdamiPressureExtrapolation(),
                                                  viscosity=viscosity,
                                                  smoothing_kernel, smoothing_length)

boundary_model_lid = BoundaryModelDummyParticles(lid.density, lid.mass,
                                                 AdamiPressureExtrapolation(),
                                                 viscosity=viscosity,
                                                 smoothing_kernel, smoothing_length)

# ==========================================================================================
# ==== Systems

fluid_system = EntropicallyDampedSPHSystem(tank.fluid, smoothing_kernel,
                                           smoothing_length,
                                           sound_speed,
                                           viscosity=viscosity,
                                           transport_velocity=TransportVelocityAdami(pressure),
                                           acceleration=(0.0, gravity))

boundary_system_tank = BoundarySPHSystem(tank.boundary, boundary_model_tank)

boundary_system_lid = BoundarySPHSystem(lid, boundary_model_lid, movement=movement)

# ==========================================================================================
# ==== Simulation

tspan = (0.0, 5.0)

semi = Semidiscretization(fluid_system, boundary_system_tank, boundary_system_lid,
                          neighborhood_search=GridNeighborhoodSearch,
                          periodic_box_min_corner=[0.0 - 4 * particle_spacing, -0.24],
                          periodic_box_max_corner=[1.0 + 4 * particle_spacing,
                              1.0 + 4 * particle_spacing])

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)

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
sol = solve(ode, RDPK3SpFSAL49((step_limiter!)=TrixiParticles.update_transport_velocity!),
            abstol=1e-6, # Default abstol is 1e-6 (may needs to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may needs to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);

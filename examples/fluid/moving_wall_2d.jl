using TrixiParticles
using OrdinaryDiffEq

gravity = -9.81

# ==========================================================================================
# ==== Fluid

particle_spacing = 0.03
# Ratio of fluid particle spacing to boundary particle spacing
beta_wall = 1
beta_tank = 1
boundary_layers = 3
boundary_layers_wall = 3

water_width = 1.0
water_height = 0.8
water_density = 1000.0

tank_width = 4.0
tank_height = 1.0

sound_speed = 10 * sqrt(9.81 * water_height)

state_equation = StateEquationCole(sound_speed, 7, water_density, 100_000.0,
                                   background_pressure=100_000.0)

viscosity = ArtificialViscosityMonaghan(0.1, 0.0)

smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

tank = RectangularTank(particle_spacing, (water_width, water_height),
                       (tank_width, tank_height), water_density,
                       n_layers=boundary_layers, spacing_ratio=beta_tank)

# ==========================================================================================
# ==== Boundary

boundary_particle_spacing = particle_spacing / beta_wall

# Move right boundary
wall_position = (tank.n_particles_per_dimension[1] + 1) * particle_spacing
n_wall_particles_y = size(tank.face_indices[2], 2) * beta_wall

wall = RectangularShape(boundary_particle_spacing,
                        (boundary_layers_wall, n_wall_particles_y),
                        (wall_position, boundary_particle_spacing), water_density)

f_y(t) = 0.0
f_x(t) = 0.5t^2

keep_moving(t) = t < 1.5

move_coordinates = MovementFunction((f_x, f_y), wall.coordinates, keep_moving)

# ==========================================================================================
# ==== Boundary models

boundary_model_tank = BoundaryModelDummyParticles(tank.boundary.density,
                                                  tank.boundary.mass, state_equation,
                                                  AdamiPressureExtrapolation(),
                                                  smoothing_kernel, smoothing_length)

boundary_model_wall = BoundaryModelDummyParticles(wall.density, wall.mass, state_equation,
                                                  AdamiPressureExtrapolation(),
                                                  smoothing_kernel, smoothing_length)

# ==========================================================================================
# ==== Systems

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, SummationDensity(), state_equation,
                                           smoothing_kernel, smoothing_length,
                                           viscosity=viscosity, acceleration=(0.0, gravity))

boundary_system_tank = BoundarySPHSystem(tank.boundary.coordinates, boundary_model_tank)

boundary_system_wall = BoundarySPHSystem(wall.coordinates, boundary_model_wall,
                                         move_coordinates=move_coordinates)

# ==========================================================================================
# ==== Simulation

tspan = (0.0, 2.0)

semi = Semidiscretization(fluid_system, boundary_system_tank, boundary_system_wall,
                          neighborhood_search=SpatialHashingSearch)

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
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-6, # Default abstol is 1e-6 (may needs to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may needs to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);

using Pixie
using OrdinaryDiffEq

particle_spacing = 0.1
# Ratio of fluid particle spacing to boundary particle spacing
beta = 3

water_width = 2.0 # x-direction
water_height = 1.0 # y-direction
water_length = 1.0 # z-direction
container_width = floor(5.366 / particle_spacing * beta) * particle_spacing / beta
container_height = 2.0
container_length = 1.0

particle_density = 1000.0

setup = RectangularTank(particle_spacing, beta,
                        water_width, water_height, water_length,
                        container_width, container_height, container_length,
                        particle_density)

# Move right boundary
reset_right_wall!(setup, container_width, wall_position=water_width)

c = 20 * sqrt(9.81 * water_height)

smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{3}()

state_equation = StateEquationCole(c, 7, 1000.0, 100000.0, background_pressure=100000.0)

particle_container = FluidParticleContainer(setup.particle_coordinates, setup.particle_velocities,
                                            setup.particle_masses, setup.particle_densities,
                                            ContinuityDensity(), state_equation,
                                            smoothing_kernel, smoothing_length,
                                            viscosity=ArtificialViscosityMonaghan(0.02, 0.0),
                                            acceleration=(0.0, -9.81, 0.0))

K = 9.81 * water_height
boundary_container = BoundaryParticleContainer(setup.boundary_coordinates, setup.boundary_masses,
                                               BoundaryModelMonaghanKajtar(K, beta, particle_spacing / beta))

semi = Semidiscretization(particle_container, boundary_container, neighborhood_search=SpatialHashingSearch)

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

alive_callback = AliveCallback(alive_interval=10)

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
sol = solve(ode, RDPK3SpFSAL49(thread=OrdinaryDiffEq.True()),
            dt=1e-5, # Small initial stepsize because the automatic choice is usually too large
            abstol=1e-5, # Higher abstol (default is 1e-6) for performance reasons
            reltol=1e-4, # Smaller reltol (default is 1e-3) to prevent boundary penetration
            save_everystep=false, callback=alive_callback);

# Move right boundary
reset_right_wall!(setup, container_width)

# Run full simulation
tspan = (0.0, 5.7 / sqrt(9.81))

# Use solution of the relaxing step as initial coordinates
u_end = Pixie.wrap_array(sol[end], 1, semi)
particle_container.initial_coordinates .= view(u_end, 1:3, :)
particle_container.initial_velocity .= view(u_end, 4:6, :)

semi = Semidiscretization(particle_container, boundary_container, neighborhood_search=SpatialHashingSearch)
ode = semidiscretize(semi, tspan)

saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:0.02:1000.0,
                                                       index=(u, t, container) -> Pixie.eachparticle(container))

callbacks = CallbackSet(alive_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
sol = solve(ode, RDPK3SpFSAL49(thread=OrdinaryDiffEq.True()),
            dt=1e-5, # Small initial stepsize because the automatic choice is usually too large
            abstol=1e-5, # Higher abstol (default is 1e-6) for performance reasons
            reltol=1e-4, # Smaller reltol (default is 1e-3) to prevent boundary penetration
            save_everystep=false, callback=callbacks);

using Pixie
using OrdinaryDiffEq

particle_spacing = 0.01
beta = 3

water_width = 2.0
water_height = 1.0
container_width = floor(5.366 / particle_spacing * beta) * particle_spacing / beta
container_height = 2.0

particle_density = 1000.0

setup = RectangularTank(particle_spacing, beta, water_width, water_height,
                        container_width, container_height, particle_density)

# Move right boundary
reset_right_wall!(setup, container_width, wall_position=water_width)

c = 20 * sqrt(9.81 * water_height)

smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()
search_radius = Pixie.compact_support(smoothing_kernel, smoothing_length)

K = 4 * 9.81 * water_height
boundary_conditions = BoundaryConditionMonaghanKajtar(setup.boundary_coordinates, setup.boundary_masses,
                                                      K, beta, particle_spacing / beta,
                                                      sound_speed=c, neighborhood_search=SpatialHashingSearch{2}(search_radius))

# Create semidiscretization
state_equation = StateEquationCole(c, 7, 1000.0, 100000.0, background_pressure=100000.0)
# state_equation = StateEquationIdealGas(10.0, 3.0, 10.0, background_pressure=10.0)

semi = SPHSemidiscretization{2}(setup.particle_masses,
                                ContinuityDensity(), state_equation,
                                smoothing_kernel, smoothing_length,
                                viscosity=ArtificialViscosityMonaghan(0.02, 0.0),
                                boundary_conditions=boundary_conditions,
                                gravity=(0.0, -9.81),
                                neighborhood_search=SpatialHashingSearch{2}(search_radius))

tspan = (0.0, 3.0)
ode = semidiscretize(semi, setup.particle_coordinates, setup.particle_velocities, setup.particle_densities, tspan)

alive_callback = AliveCallback(alive_interval=100)

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
sol = solve(ode, RDPK3SpFSAL49(thread=OrdinaryDiffEq.True()),
            dt=1e-4, # Initial guess of the time step to prevent too large guesses
            abstol=1.0e-4, reltol=1.0e-4, # Tighter tolerance to prevent instabilities, use 2e-5 for spacing 0.004
            save_everystep=false, callback=alive_callback);

# Move right boundary
reset_right_wall!(setup, container_width)

# Run full simulation
tspan = (0.0, 5.7 / sqrt(9.81))
ode = semidiscretize(semi, view(sol[end], 1:2, :), view(sol[end], 3:4, :), view(sol[end], 5, :), tspan)

saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:0.02:20.0,
                                                       index=(u, t, integrator) -> Pixie.eachparticle(integrator.p))

callbacks = CallbackSet(alive_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
sol = solve(ode, RDPK3SpFSAL49(thread=OrdinaryDiffEq.True()),
            dt=1e-4, # Initial guess of the time step to prevent too large guesses
            abstol=1.0e-4, reltol=1.0e-4, # Tighter tolerance to prevent instabilities
            save_everystep=false, callback=callbacks);

using Pixie
using .OrdinaryDiffEq

particle_spacing = 0.02
beta = 3

water_width = 2.0
water_height = 1.0
container_width = floor(5.366 / particle_spacing * beta) * particle_spacing / beta
container_height = 2.0

particle_density = 1000.0

setup = RectangularTank(particle_spacing, beta, water_width, water_height,
                        container_width, container_height, particle_density,
                        n_layers=1)

# Move right boundary
reset_right_wall!(setup, container_width, wall_position=water_width)

c = 20 * sqrt(9.81 * water_height)

smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

state_equation = StateEquationCole(c, 7, 1000.0, 100000.0, background_pressure=100000.0)
# state_equation = StateEquationIdealGas(10.0, 3.0, 10.0, background_pressure=10.0)

particle_container = FluidParticleContainer(setup.particle_coordinates, setup.particle_velocities,
                                            setup.particle_masses, setup.particle_densities,
                                            ContinuityDensity(), state_equation,
                                            smoothing_kernel, smoothing_length,
                                            viscosity=ArtificialViscosityMonaghan(0.02, 0.0),
                                            acceleration=(0.0, -9.81))

K = 4 * 9.81 * water_height
boundary_container = BoundaryParticleContainer(setup.boundary_coordinates, setup.boundary_masses,
                                               BoundaryModelMonaghanKajtar(K, beta, particle_spacing / beta))

semi = Semidiscretization(particle_container, boundary_container, neighborhood_search=SpatialHashingSearch)

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

alive_callback = AliveCallback(alive_interval=100)

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
sol = solve(ode, RDPK3SpFSAL35(thread=OrdinaryDiffEq.True()),
            dt=1e-4, # Initial guess of the time step to prevent too large guesses
            abstol=1.0e-4, reltol=1.0e-4, # Tighter tolerance to prevent instabilities, use 2e-5 for spacing 0.004
            save_everystep=false, callback=alive_callback);

# Move right boundary
reset_right_wall!(setup, container_width)

# Run full simulation
tspan = (0.0, 5.7 / sqrt(9.81))

# Use solution of the relaxing step as initial coordinates
u_end = Pixie.wrap_array(sol[end], 1, semi)
particle_container.initial_coordinates .= view(u_end, 1:2, :)
particle_container.initial_velocity .= view(u_end, 3:4, :)

semi = Semidiscretization(particle_container, boundary_container, neighborhood_search=SpatialHashingSearch)
ode = semidiscretize(semi, tspan)

saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:0.02:20.0,
                                                       index=(u, t, container) -> Pixie.eachparticle(container))

callbacks = CallbackSet(alive_callback, saving_callback)


# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
sol = solve(ode, RDPK3SpFSAL35(thread=OrdinaryDiffEq.True()),
            dt=1e-4, # Initial guess of the time step to prevent too large guesses
            abstol=1.0e-4, reltol=1.0e-4, # Tighter tolerance to prevent instabilities
            save_everystep=false, callback=callbacks);

# save to vtk
pixie2vtk(saved_values, boundary_container)
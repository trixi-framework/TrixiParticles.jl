using Pixie
using OrdinaryDiffEq

water_width = 0.5
water_height = 1.0
container_width = 4.0
container_height = 2.0
particle_density = 1000.0
particle_spacing = 0.02
beta = 3

setup = RectangularTank(particle_spacing, beta, water_width, water_height,
                        container_width, container_height, particle_density, n_layers=1)

# Move water column
for i in axes(setup.particle_coordinates, 2)
    setup.particle_coordinates[:, i] .+= [0.5 * container_width - 0.5 * water_width, 0.2]
end

c = 10 * sqrt(9.81 * water_height)
state_equation = StateEquationCole(c, 7, 1000.0, 100000.0, background_pressure=100000.0)

smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

# Create semidiscretization
particle_container = FluidParticleContainer(setup.particle_coordinates, setup.particle_velocities,
                                            setup.particle_masses, setup.particle_densities,
                                            ContinuityDensity(), state_equation,
                                            smoothing_kernel, smoothing_length,
                                            viscosity=ArtificialViscosityMonaghan(0.02, 0.0),
                                            acceleration=(0.0, -9.81))

K = 9.81 * water_height
boundary_container = BoundaryParticleContainer(setup.boundary_coordinates, setup.boundary_masses,
                                               BoundaryModelMonaghanKajtar(K, beta, particle_spacing / beta))

semi = Semidiscretization(particle_container, boundary_container, neighborhood_search=SpatialHashingSearch)

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

alive_callback = AliveCallback(alive_interval=10)
saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:0.02:1000.0)

callbacks = CallbackSet(alive_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
sol = solve(ode, RDPK3SpFSAL49(thread=OrdinaryDiffEq.True()),
            dt=1e-5, # Small initial stepsize because the automatic choice is usually too large
            abstol=1e-5, # Higher abstol (default is 1e-6) for performance reasons
            reltol=5e-4, # Smaller reltol (default is 1e-3) to prevent boundary penetration
            save_everystep=false, callback=callbacks);

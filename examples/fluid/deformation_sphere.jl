using Pixie
using OrdinaryDiffEq

particle_spacing = 0.05

water_width = 1.0
water_height = 1.0
water_density = 1000.0

setup = RectangularShape(particle_spacing, (0.0, 0.0), (water_width, water_height), density = water_density)

c = 10 * sqrt(9.81 * water_height)
state_equation = StateEquationCole(c, 7, water_density, 100000.0,
	background_pressure = 100000.0)

smoothing_length = 2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

# Create semidiscretization
particle_container = FluidParticleContainer(setup.coordinates,
	setup.velocity,
	setup.masses, setup.radius, setup.densities,
	ContinuityDensity(), state_equation,
	smoothing_kernel, smoothing_length, water_density,
	viscosity = ArtificialViscosityMonaghan(0.02,
		0.0),
	acceleration = (0.0, 0.0),
	surface_tension = CohesionForceAkinci(2e-5))

semi = Semidiscretization(particle_container,
	neighborhood_search = SpatialHashingSearch,
	damping_coefficient = 0.0)

tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 10)
saved_values, saving_callback = SolutionSavingCallback(saveat = 0.0:0.02:1000.0)

callbacks = CallbackSet(summary_callback, alive_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Enable threading of the RK method for better performance on multiple threads.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so with Monaghan-Kajtar BC because forces
# become extremely large when fluid particles are very close to boundary particles,
# and the time integration method interprets this as an instability.
sol = solve(ode, RDPK3SpFSAL49(),
	abstol = 1e-10, # Default abstol is 1e-6 (may needs to be tuned to prevent boundary penetration)
	reltol = 1e-8, # Default reltol is 1e-3 (may needs to be tuned to prevent boundary penetration)
	dtmax = 1e-3, # Limit stepsize to prevent crashing
	save_everystep = false, callback = callbacks);

# Print the timer summary
summary_callback()

# activate to save to vtk
pixie2vtk(saved_values)

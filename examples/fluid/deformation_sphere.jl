using Pixie
using OrdinaryDiffEq

particle_spacing = 0.2

water_width = 1.0
water_height = 1.0
water_density = 1000.0
#water_viscosity = 10


# relax=0.99 630 max press
setup = RectangularShape(particle_spacing, (0.0, 0.0), (water_width, water_height), density = water_density)

c = 10 * sqrt(9.81 * water_height)
state_equation = StateEquationCole(c, 7, water_density, 100000.0,
	background_pressure = 100000.0)

smoothing_length = 4.0 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()
# Monaghan 2005
#alpha = 10 * water_viscosity/(water_density * smoothing_length)

# Create semidiscretization
particle_container = FluidParticleContainer(setup.coordinates,
	setup.velocity,
	setup.masses, setup.radius, setup.densities,
	SummationDensity(), state_equation,
	smoothing_kernel, smoothing_length, water_density,
	viscosity = ArtificialViscosityMonaghan(1.0,
		2.0),
	acceleration = (0.0, 0.0),
	surface_tension = SurfaceTensionAkinci(surface_tension_coefficient=0.2, support_length=2.0*particle_spacing))

# particle_container = FluidParticleContainer(setup.coordinates,
# 	setup.velocity,
# 	setup.masses, setup.radius, setup.densities,
# 	ContinuityDensity(), state_equation,
# 	smoothing_kernel, smoothing_length, water_density,
# 	viscosity = ArtificialViscosityMonaghan(1.0,
# 		2.0),
# 	acceleration = (0.0, 0.0),
# 	surface_tension = SurfaceTensionAkinci(surface_tension_coefficient=0.5, support_length=2.0*particle_spacing))

semi = Semidiscretization(particle_container,
	neighborhood_search = SpatialHashingSearch,
	damping_coefficient = 0.0)

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 100)
saved_values, saving_callback = SolutionSavingCallback(saveat = 0.0:0.02:1000.0, index=(u, t, container) -> Pixie.eachparticle(container))

callbacks = CallbackSet(summary_callback, alive_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Enable threading of the RK method for better performance on multiple threads.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so with Monaghan-Kajtar BC because forces
# become extremely large when fluid particles are very close to boundary particles,
# and the time integration method interprets this as an instability.
# sol = solve(ode, RDPK3SpFSAL49(),
# 	abstol = 1e-8, # Default abstol is 1e-6 (may needs to be tuned to prevent boundary penetration)
# 	reltol = 1e-6, # Default reltol is 1e-3 (may needs to be tuned to prevent boundary penetration)
# 	dtmax = 5e-5, # Limit stepsize to prevent crashing
# 	save_everystep = false, callback = callbacks);


sol = solve(ode, TRBDF2(autodiff=false),
abstol = 1e-8, # Default abstol is 1e-6 (may needs to be tuned to prevent boundary penetration)
reltol = 1e-6, # Default reltol is 1e-3 (may needs to be tuned to prevent boundary penetration)
dtmax = 5e-5, # Limit stepsize to prevent crashing
save_everystep = false, callback = callbacks);

# Print the timer summary
summary_callback()

# activate to save to vtk
pixie2vtk(saved_values)

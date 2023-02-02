using Pixie
using OrdinaryDiffEq

particle_spacing = 0.2

water_width = 1.0
water_height = 1.0
water_depth = 1.0
water_density = 1000.0

setup = RectangularShape(particle_spacing, (5, 5, 5), (0.0, 0.0, 0.0),
                         density=water_density)

c = 10 * sqrt(9.81 * water_height)
state_equation = StateEquationCole(c, 7, water_density, 100000.0,
                                   background_pressure=100000.0)

smoothing_length = 2.0 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

# Create semidiscretization
particle_container = FluidParticleContainer(setup.coordinates,
                                            setup.velocity,
                                            setup.masses, setup.radius,
                                            SummationDensity(), state_equation,
                                            smoothing_kernel, smoothing_length,
                                            water_density,
                                            viscosity=ArtificialViscosityMonaghan(1.0,
                                                                                  2.0),
                                            acceleration=(0.0, 0.0, 0.0),
                                            surface_tension=SurfaceTensionAkinci(surface_tension_coefficient=0.05))

semi = Semidiscretization(particle_container,
                          neighborhood_search=SpatialHashingSearch,
                          damping_coefficient=0.0)

tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=100)
saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:0.02:1000.0,
                                                       index=(v, u, t, container) -> Pixie.eachparticle(container))

callbacks = CallbackSet(summary_callback, alive_callback, saving_callback)

sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-5, # Default abstol is 1e-6 (may needs to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may needs to be tuned to prevent boundary penetration)
            dtmax=5e-3, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()

# activate to save to vtk
pixie2vtk(saved_values)

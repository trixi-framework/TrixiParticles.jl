using Pixie
using OrdinaryDiffEq

# ==========================================================================================
# ==== Reference Values

gravity = 9.81
atmospheric_pressure = 100000.0
incompressible_gamma = 7
ambient_temperature = 293.15


# ==========================================================================================
# ==== Fluid

water_width = 1.0
water_height = 1.0
water_density = 1000.0


c = 10 * sqrt(gravity * water_height)
state_equation = StateEquationCole(c, incompressible_gamma, water_density, atmospheric_pressure,
                                   background_pressure=atmospheric_pressure)

water_at_rest = State(water_density, atmospheric_pressure, ambient_temperature)

# ==========================================================================================
# ==== Particle Setup

particle_spacing = 0.2

smoothing_length = 4.0 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

setup = RectangularShape(particle_spacing, (5, 5), (0.0, 0.0),  density=water_density)

# ==========================================================================================
# ==== Containers

particle_container = FluidParticleContainer(setup,
                                            SummationDensity(), state_equation,
                                            smoothing_kernel, smoothing_length,
                                            water_at_rest,
                                            viscosity=ArtificialViscosityMonaghan(1.0,
                                                                                  2.0),
                                            acceleration=(0.0, 0.0),
                                            surface_tension=SurfaceTensionAkinci(surface_tension_coefficient=0.2),
                                            store_options=StoreAll())

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(particle_container,
                          neighborhood_search=SpatialHashingSearch,
                          damping_coefficient=0.0)

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=100)
saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:0.02:1000.0,
                                                       index=(v, u, t, container) -> Pixie.eachparticle(container))

callbacks = CallbackSet(summary_callback, alive_callback, saving_callback)

sol = solve(ode, TRBDF2(autodiff=false),
            abstol=1e-8, # Default abstol is 1e-6 (may needs to be tuned to prevent boundary penetration)
            reltol=1e-6, # Default reltol is 1e-3 (may needs to be tuned to prevent boundary penetration)
            dtmax=5e-5, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()

# activate to save to vtk
# pixie2vtk(saved_values)

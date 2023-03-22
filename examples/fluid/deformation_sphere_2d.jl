using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Reference Values

gravity = 9.81
atmospheric_pressure = 1E5
incompressible_gamma = 7.0

# ==========================================================================================
# ==== Fluid
water_density = 1000.0

c = 10 * sqrt(gravity)
state_equation = StateEquationCole(c, incompressible_gamma, water_density,
                                   atmospheric_pressure,
                                   background_pressure=atmospheric_pressure)

# ==========================================================================================
# ==== Particle Setup

particle_spacing = 0.2

smoothing_length = 4.0 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

setup = RectangularShape(particle_spacing, (5, 5), (0.0, 0.0), density=water_density)

# ==========================================================================================
# ==== Containers

particle_container = FluidParticleContainer(setup,
                                            SummationDensity(), state_equation,
                                            smoothing_kernel, smoothing_length,
                                            water_density,
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

info_callback = InfoCallback(interval=1000)
saving_callback = SolutionSavingCallback(dt=0.02)

callbacks = CallbackSet(info_callback, saving_callback)

sol = solve(ode, TRBDF2(autodiff=false),
            abstol=1e-8, # Default abstol is 1e-6 (may needs to be tuned to prevent boundary penetration)
            reltol=1e-6, # Default reltol is 1e-3 (may needs to be tuned to prevent boundary penetration)
            dtmax=5e-5, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);

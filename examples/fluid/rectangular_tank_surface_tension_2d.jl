using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Reference Values

gravity = 9.81
atmospheric_pressure = 100000.0
incompressible_gamma = 7
ambient_temperature = 293.15

# ==========================================================================================
# ==== Fluid
water_width = 2.0
water_height = 0.9
water_density = 1000.0

water_at_rest = State(water_density, atmospheric_pressure, ambient_temperature)

# ==========================================================================================
# ==== Particle Setup

particle_spacing = 0.02
beta = 1

tank_width = 2.0
tank_height = 1.0

setup = RectangularTank(particle_spacing, (water_width, water_height),
                        (tank_width, tank_height), water_density, n_layers=3,
                        spacing_ratio=beta)

c = 10 * sqrt(gravity * water_height)
state_equation = StateEquationCole(c, incompressible_gamma, water_density,
                                   atmospheric_pressure,
                                   background_pressure=atmospheric_pressure)

smoothing_length = 2.0 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

# ==========================================================================================
# ==== Containers
particle_container = FluidParticleContainer(setup,
                                            SummationDensity(), state_equation,
                                            smoothing_kernel, smoothing_length,
                                            water_at_rest,
                                            viscosity=ArtificialViscosityMonaghan(1.0,
                                                                                  2.0),
                                            acceleration=(0.0, -9.81),
                                            surface_tension=SurfaceTensionAkinci(surface_tension_coefficient=0.001))

# ==========================================================================================
# ==== Boundary models

#boundary_densities = water_density * ones(size(setup.boundary_masses))
boundary_model = BoundaryModelDummyParticles(setup.boundary_densities,
                                             setup.boundary_masses,
                                             state_equation,
                                             AdamiPressureExtrapolation(), smoothing_kernel,
                                             smoothing_length)

boundary_container = BoundaryParticleContainer(setup.boundary_coordinates,
                                               boundary_model)

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(particle_container, boundary_container,
                          neighborhood_search=SpatialHashingSearch,
                          damping_coefficient=1e-5)

tspan = (0.0, 2.0)
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

using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.02
solid_particle_spacing = fluid_particle_spacing

# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
boundary_layers = 3
spacing_ratio = 1

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 2.0)

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (2.0, 0.9)
tank_size = (2.0, 1.0)

fluid_density = 1000.0
atmospheric_pressure = 100000.0
sound_speed = 10 * sqrt(gravity * initial_fluid_size[2])
state_equation = StateEquationCole(sound_speed, 7, fluid_density, atmospheric_pressure,
                                   background_pressure=atmospheric_pressure)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       faces=(true, true, true, false),
                       acceleration=(0.0, -gravity), state_equation=state_equation)

sphere1_radius = 0.3
sphere2_radius = 0.2
sphere1_density = 500.0
sphere2_density = 1100.0

# Young's modulus and Poisson ratio
sphere1_E = 7e4
sphere2_E = 1e5
nu = 0.0

sphere1 = SphereShape(solid_particle_spacing, sphere1_radius, (0.5, 1.6),
                      sphere1_density)
sphere2 = SphereShape(solid_particle_spacing, sphere2_radius, (1.5, 1.6),
                      sphere2_density)

# ==========================================================================================
# ==== Fluid
fluid_smoothing_length = 1.2 * fluid_particle_spacing
fluid_smoothing_kernel = SchoenbergCubicSplineKernel{2}()

fluid_density_calculator = ContinuityDensity()
viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, fluid_smoothing_kernel,
                                           fluid_smoothing_length, viscosity=viscosity,
                                           acceleration=(0.0, -gravity))

# ==========================================================================================
# ==== Boundary
boundary_density_calculator = AdamiPressureExtrapolation()
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             fluid_smoothing_kernel, fluid_smoothing_length)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Solid
solid_smoothing_length = sqrt(2) * solid_particle_spacing
solid_smoothing_kernel = SchoenbergCubicSplineKernel{2}()

# For the FSI we need the hydrodynamic masses and densities in the solid boundary model
hydrodynamic_densites_1 = fluid_density * ones(size(sphere1.density))
hydrodynamic_masses_1 = hydrodynamic_densites_1 * solid_particle_spacing^2

solid_boundary_model_1 = BoundaryModelDummyParticles(hydrodynamic_densites_1,
                                                     hydrodynamic_masses_1,
                                                     state_equation=state_equation,
                                                     boundary_density_calculator,
                                                     fluid_smoothing_kernel,
                                                     fluid_smoothing_length)

hydrodynamic_densites_2 = fluid_density * ones(size(sphere2.density))
hydrodynamic_masses_2 = hydrodynamic_densites_2 * solid_particle_spacing^2

solid_boundary_model_2 = BoundaryModelDummyParticles(hydrodynamic_densites_2,
                                                     hydrodynamic_masses_2,
                                                     state_equation=state_equation,
                                                     boundary_density_calculator,
                                                     fluid_smoothing_kernel,
                                                     fluid_smoothing_length)

solid_system_1 = TotalLagrangianSPHSystem(sphere1,
                                          solid_smoothing_kernel, solid_smoothing_length,
                                          sphere1_E, nu,
                                          acceleration=(0.0, gravity),
                                          solid_boundary_model_1,
                                          penalty_force=PenaltyForceGanzenmueller(alpha=0.3))

solid_system_2 = TotalLagrangianSPHSystem(sphere2,
                                          solid_smoothing_kernel, solid_smoothing_length,
                                          sphere2_E, nu,
                                          acceleration=(0.0, gravity),
                                          solid_boundary_model_2,
                                          penalty_force=PenaltyForceGanzenmueller(alpha=0.3))

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, boundary_system, solid_system_1, solid_system_2,
                          neighborhood_search=GridNeighborhoodSearch)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=10)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so because forces become extremely large when
# fluid particles are very close to boundary particles, and the time integration method
# interprets this as an instability.
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-3, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);

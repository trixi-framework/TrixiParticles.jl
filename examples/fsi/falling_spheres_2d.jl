# In this example, two solid spheres of different densities fall into a tank of water.
# Note that the solids don't interact with boundaries (yet), so that the sphere with larger
# density is going to fall out of the tank.

using TrixiParticles
using OrdinaryDiffEq

gravity = -9.81

# ==========================================================================================
# ==== Fluid

fluid_particle_spacing = 0.02

# Ratio of fluid particle spacing to boundary particle spacing
beta = 1
boundary_layers = 3

water_width = 2.0
water_height = 0.9
water_density = 1000.0

tank_width = 2.0
tank_height = 1.0

sound_speed = 10 * sqrt(9.81 * water_height)
state_equation = StateEquationCole(sound_speed, 7, water_density, 100000.0,
                                   background_pressure=100000.0)

fluid_smoothing_length = 1.2 * fluid_particle_spacing
fluid_smoothing_kernel = SchoenbergCubicSplineKernel{2}()

viscosity = ArtificialViscosityMonaghan(0.02, 0.0)

tank = RectangularTank(fluid_particle_spacing, (water_width, water_height),
                       (tank_width, tank_height), water_density,
                       n_layers=boundary_layers, spacing_ratio=beta,
                       faces=(true, true, true, false),
                       acceleration=(0.0, gravity), state_equation=state_equation)

# ==========================================================================================
# ==== Solid

solid_radius_1 = 0.3
solid_radius_2 = 0.2
solid_density_1 = 500.0
solid_density_2 = 1100.0
solid_particle_spacing = fluid_particle_spacing

solid_smoothing_length = sqrt(2) * solid_particle_spacing
solid_smoothing_kernel = SchoenbergCubicSplineKernel{2}()

# Young's modulus and Poisson ratio
E1 = 7e4
E2 = 1e5
nu = 0.0

sphere_1 = SphereShape(solid_particle_spacing, solid_radius_1, (0.5, 1.6),
                       solid_density_1)
sphere_2 = SphereShape(solid_particle_spacing, solid_radius_2, (1.5, 1.6),
                       solid_density_2)

# ==========================================================================================
# ==== Boundary models

boundary_model = BoundaryModelDummyParticles(tank.boundary.density,
                                             tank.boundary.mass,
                                             state_equation=state_equation,
                                             AdamiPressureExtrapolation(),
                                             fluid_smoothing_kernel,
                                             fluid_smoothing_length)

# For the FSI we need the hydrodynamic masses and densities in the solid boundary model
hydrodynamic_densites_1 = water_density * ones(size(sphere_1.density))
hydrodynamic_masses_1 = hydrodynamic_densites_1 * solid_particle_spacing^2

solid_boundary_model_1 = BoundaryModelDummyParticles(hydrodynamic_densites_1,
                                                     hydrodynamic_masses_1,
                                                     state_equation=state_equation,
                                                     AdamiPressureExtrapolation(),
                                                     fluid_smoothing_kernel,
                                                     fluid_smoothing_length)

hydrodynamic_densites_2 = water_density * ones(size(sphere_2.density))
hydrodynamic_masses_2 = hydrodynamic_densites_2 * solid_particle_spacing^2

solid_boundary_model_2 = BoundaryModelDummyParticles(hydrodynamic_densites_2,
                                                     hydrodynamic_masses_2,
                                                     state_equation=state_equation,
                                                     AdamiPressureExtrapolation(),
                                                     fluid_smoothing_kernel,
                                                     fluid_smoothing_length)

# ==========================================================================================
# ==== Systems

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, ContinuityDensity(), state_equation,
                                           fluid_smoothing_kernel,
                                           fluid_smoothing_length,
                                           viscosity=viscosity,
                                           acceleration=(0.0, gravity))

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

solid_system_1 = TotalLagrangianSPHSystem(sphere_1,
                                          solid_smoothing_kernel, solid_smoothing_length,
                                          E1, nu,
                                          acceleration=(0.0, gravity),
                                          solid_boundary_model_1,
                                          penalty_force=PenaltyForceGanzenmueller(alpha=0.3))

solid_system_2 = TotalLagrangianSPHSystem(sphere_2,
                                          solid_smoothing_kernel, solid_smoothing_length,
                                          E2, nu,
                                          acceleration=(0.0, gravity),
                                          solid_boundary_model_2,
                                          penalty_force=PenaltyForceGanzenmueller(alpha=0.3))

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(fluid_system, boundary_system, solid_system_1,
                          solid_system_2,
                          neighborhood_search=GridNeighborhoodSearch)

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=10)
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
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-3, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);

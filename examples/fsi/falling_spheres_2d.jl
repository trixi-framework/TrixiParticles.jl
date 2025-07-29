# ==========================================================================================
# 2D Falling Spheres in Fluid (FSI) - Base Setup
#
# This file provides a base setup for simulating one or two elastic spheres
# falling into a fluid in a tank.
# ==========================================================================================

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
tspan = (0.0, 1.0)

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (2.0, 0.9)
tank_size = (2.0, 1.0)

fluid_density = 1000.0
sound_speed = 10 * sqrt(gravity * initial_fluid_size[2])
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1)

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

sphere1_center = (0.5, 1.6)
sphere2_center = (1.5, 1.6)
sphere1 = SphereShape(solid_particle_spacing, sphere1_radius, sphere1_center,
                      sphere1_density, sphere_type=VoxelSphere())
sphere2 = SphereShape(solid_particle_spacing, sphere2_radius, sphere2_center,
                      sphere2_density, sphere_type=VoxelSphere())

# ==========================================================================================
# ==== Fluid
fluid_smoothing_length = 1.5 * fluid_particle_spacing
fluid_smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()
viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)
density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, fluid_smoothing_kernel,
                                           fluid_smoothing_length, viscosity=viscosity,
                                           density_diffusion=density_diffusion,
                                           acceleration=(0.0, -gravity))

# ==========================================================================================
# ==== Boundary
boundary_density_calculator = BernoulliPressureExtrapolation()
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             fluid_smoothing_kernel, fluid_smoothing_length)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Solid
solid_smoothing_length = sqrt(2) * solid_particle_spacing
solid_smoothing_kernel = WendlandC2Kernel{2}()

# For the FSI we need the hydrodynamic masses and densities in the solid boundary model
hydrodynamic_densites_1 = fluid_density * ones(size(sphere1.density))
hydrodynamic_masses_1 = hydrodynamic_densites_1 * solid_particle_spacing^ndims(fluid_system)

solid_boundary_model_1 = BoundaryModelDummyParticles(hydrodynamic_densites_1,
                                                     hydrodynamic_masses_1,
                                                     state_equation=state_equation,
                                                     boundary_density_calculator,
                                                     fluid_smoothing_kernel,
                                                     fluid_smoothing_length)

hydrodynamic_densites_2 = fluid_density * ones(size(sphere2.density))
hydrodynamic_masses_2 = hydrodynamic_densites_2 * solid_particle_spacing^ndims(fluid_system)

solid_boundary_model_2 = BoundaryModelDummyParticles(hydrodynamic_densites_2,
                                                     hydrodynamic_masses_2,
                                                     state_equation=state_equation,
                                                     boundary_density_calculator,
                                                     fluid_smoothing_kernel,
                                                     fluid_smoothing_length)

solid_system_1 = TotalLagrangianSPHSystem(sphere1,
                                          solid_smoothing_kernel, solid_smoothing_length,
                                          sphere1_E, nu,
                                          acceleration=(0.0, -gravity),
                                          boundary_model=solid_boundary_model_1,
                                          penalty_force=PenaltyForceGanzenmueller(alpha=0.3))

solid_system_2 = TotalLagrangianSPHSystem(sphere2,
                                          solid_smoothing_kernel, solid_smoothing_length,
                                          sphere2_E, nu,
                                          acceleration=(0.0, -gravity),
                                          boundary_model=solid_boundary_model_2,
                                          penalty_force=PenaltyForceGanzenmueller(alpha=0.3))

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, boundary_system, solid_system_1, solid_system_2)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
saving_callback = SolutionSavingCallback(dt=0.02, output_directory="out", prefix="",
                                         write_meta_data=true)

callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-6, # Default abstol is 1e-6
            reltol=1e-3, # Default reltol is 1e-3
            save_everystep=false, callback=callbacks);

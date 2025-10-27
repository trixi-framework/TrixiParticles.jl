# ==========================================================================================
# 2D Dam Break Simulation (δ-SPH Model)
#
# Based on:
#   S. Marrone, M. Antuono, A. Colagrossi, G. Colicchio, D. le Touzé, G. Graziani.
#   "δ-SPH model for simulating violent impact flows".
#   Computer Methods in Applied Mechanics and Engineering, Volume 200, Issues 13–16 (2011),
#   pages 1526–1542.
#   https://doi.org/10.1016/J.CMA.2010.12.016
#
# This example sets up a 2D dam break simulation using a weakly compressible SPH scheme
# with a δ-SPH formulation for density calculation.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# Size parameters
H = 0.6
W = 2 * H

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = H / 40

# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
boundary_layers = 4
spacing_ratio = 1

boundary_particle_spacing = fluid_particle_spacing / spacing_ratio

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81

tspan = (0.0, 5.7 / sqrt(gravity / H))

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (W, H)
tank_size = (floor(5.366 * H / boundary_particle_spacing) * boundary_particle_spacing, 4.0)

fluid_density = 1000.0
sound_speed = 20 * sqrt(gravity * H)
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1, clip_negative_pressure=false)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       acceleration=(0.0, -gravity), state_equation=state_equation)

# ==========================================================================================
# ==== Fluid
smoothing_length = 2 * fluid_particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()

alpha = 0.02
viscosity_fluid = ArtificialViscosityMonaghan(alpha=alpha, beta=0.0)

# The density diffusion model by Molteni and Colagrossi shows unphysical effects at the
# free surface in long-running simulations, but is significantly faster than the model
# by Antuono. This simulation is short enough to use the faster model.
density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)
# density_diffusion = DensityDiffusionAntuono(tank.fluid, delta=0.1)

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length, viscosity=viscosity_fluid,
                                           density_diffusion=density_diffusion,
                                           acceleration=(0.0, -gravity), correction=nothing,
                                           surface_tension=nothing,
                                           reference_particle_spacing=0)

# ==========================================================================================
# ==== Boundary
boundary_density_calculator = AdamiPressureExtrapolation()
viscosity_wall = nothing
# For a no-slip boundary condition, define a wall viscosity:
# viscosity_wall = viscosity_fluid
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length,
                                             correction=nothing,
                                             reference_particle_spacing=0,
                                             viscosity=viscosity_wall)

boundary_system = WallBoundarySystem(tank.boundary, boundary_model,
                                     adhesion_coefficient=0.0)

# ==========================================================================================
# ==== Simulation
# `nothing` will automatically choose the best update strategy. This is only to be able
# to change this with `trixi_include`.
semi = Semidiscretization(fluid_system, boundary_system,
                          neighborhood_search=GridNeighborhoodSearch{2}(update_strategy=nothing),
                          parallelization_backend=PolyesterBackend())
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)

solution_prefix = ""
saving_callback = SolutionSavingCallback(dt=0.02, prefix=solution_prefix)

# This can be overwritten with `trixi_include`
extra_callback = nothing
extra_callback2 = nothing

use_reinit = false
density_reinit_cb = use_reinit ?
                    DensityReinitializationCallback(semi.systems[1], interval=10) :
                    nothing
stepsize_callback = StepsizeCallback(cfl=0.9)

callbacks = CallbackSet(info_callback, saving_callback, stepsize_callback, extra_callback,
                        extra_callback2, density_reinit_cb)

time_integration_scheme = CarpenterKennedy2N54(williamson_condition=false)
sol = solve(ode, time_integration_scheme,
            dt=1.0, # This is overwritten by the stepsize callback
            save_everystep=false, callback=callbacks);

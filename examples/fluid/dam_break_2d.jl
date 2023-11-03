# 2D dam break simulation based on
#
# S. Marrone, M. Antuono, A. Colagrossi, G. Colicchio, D. le Touzé, G. Graziani.
# "δ-SPH model for simulating violent impact flows".
# In: Computer Methods in Applied Mechanics and Engineering, Volume 200, Issues 13–16 (2011), pages 1526–1542.
# https://doi.org/10.1016/J.CMA.2010.12.016

using TrixiParticles
using OrdinaryDiffEq

# Constants
gravity = 9.81
atmospheric_pressure = 100000.0
fluid_density = 1000.0

# Simulation settings
fluid_particle_spacing = 0.02
smoothing_length = 1.2 * fluid_particle_spacing
boundary_layers = 4
# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
spacing_ratio = 1
boundary_particle_spacing = fluid_particle_spacing / spacing_ratio
output_dt = 0.02
file_prefix = ""
tspan = (0.0, 5.7 / sqrt(gravity))

# Model settings
fluid_density_calculator = ContinuityDensity()
boundary_density_calculator = AdamiPressureExtrapolation()
use_reinit = false

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (2.0, 1.0)
tank_size = (floor(5.366 / boundary_particle_spacing) * boundary_particle_spacing, 4.0)

# ==========================================================================================
# ==== Fluid
sound_speed = 20 * sqrt(gravity * initial_fluid_size[2])

state_equation = StateEquationCole(sound_speed, 7, fluid_density, atmospheric_pressure,
                                   background_pressure=atmospheric_pressure)

smoothing_kernel = SchoenbergCubicSplineKernel{2}()

viscosity = ArtificialViscosityMonaghan(0.02, 0.0)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       acceleration=(0.0, -gravity), state_equation=state_equation)

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length, viscosity=viscosity,
                                           acceleration=(0.0, -gravity), correction=nothing)

# ==========================================================================================
# ==== Boundary models
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length)

# K = gravity * initial_fluid_size[2]
# boundary_model = BoundaryModelMonaghanKajtar(K, beta, fluid_particle_spacing / beta,
#                                              tank.boundary.mass)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(fluid_system, boundary_system,
                          neighborhood_search=GridNeighborhoodSearch)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=output_dt, prefix=file_prefix)
density_reinit_cb = use_reinit ? DensityReinitializationCallback(semi.systems[1], dt=0.01) :
                    nothing

callbacks = CallbackSet(info_callback, saving_callback, density_reinit_cb)

# See above for an explanation of the parameter choice
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-5, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);

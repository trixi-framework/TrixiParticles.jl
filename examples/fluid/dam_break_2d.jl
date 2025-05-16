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

# ------------------------------------------------------------------------------
# Physical and Resolution Parameters
# ------------------------------------------------------------------------------

initial_water_height = 0.6
initial_water_width = 2 * initial_water_height

# Particle spacing or resolution of the simulation
fluid_particle_spacing = initial_water_height / 40

# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
boundary_layers = 4 # Number of boundary particle layers
spacing_ratio = 1   # Ratio of fluid particle spacing to boundary particle spacing

boundary_particle_spacing = fluid_particle_spacing / spacing_ratio

# ------------------------------------------------------------------------------
# Experiment Setup:
# ------------------------------------------------------------------------------

# Gravitational acceleration
gravity = 9.81

# Simulation time span
tspan = (0.0, 5.7 / sqrt(gravity)) # Normalized time based on Marrone et al.

# Define initial fluid domain and tank dimensions
initial_fluid_size = (initial_water_width, initial_water_height)
# Tank width is chosen to match the reference paper (5.366 times initial_water_height)
# and floored to be a multiple of boundary_particle_spacing. Tank height is set to 4.0 (arbitrary, large enough).
tank_size = (floor(5.366 * initial_water_height / boundary_particle_spacing) *
             boundary_particle_spacing, 4.0)

# Fluid properties
fluid_density = 1000.0 # Reference density of water

# Speed of sound: Should be high enough to ensure low compressibility,
sound_speed = 20 * sqrt(gravity * initial_water_height)

state_equation = StateEquationCole(sound_speed=sound_speed,
                                   reference_density=fluid_density,
                                   exponent=1, # For weakly compressible assumption with Tait-like equation
                                   clip_negative_pressure=false)

# Setup for the rectangular tank with fluid and boundary particles
tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers,
                       spacing_ratio=spacing_ratio,
                       acceleration=(0.0, -gravity),
                       state_equation=state_equation)

# ------------------------------------------------------------------------------
# Fluid System Setup
# ------------------------------------------------------------------------------

# SPH kernel and smoothing length
smoothing_length = 1.75 * fluid_particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

# Density calculation method and viscosity model
fluid_density_calculator = ContinuityDensity()
alpha = 0.02
viscosity = ArtificialViscosityMonaghan(alpha=alpha, beta=0.0)
# Alternative viscosity models:
# nu = alpha * smoothing_length * sound_speed / 8
# viscosity = ViscosityMorris(nu=nu)
# viscosity = ViscosityAdami(nu=nu)

density_diffusion = DensityDiffusionAntuono(tank.fluid, delta=0.1)
# Alternatively the density diffusion model by Molteni & Colagrossi can be used,
# which will run faster but will be less accurate.
# density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation,
                                           smoothing_kernel, smoothing_length,
                                           viscosity=viscosity,
                                           density_diffusion=density_diffusion,
                                           acceleration=(0.0, -gravity),
                                           correction=nothing,
                                           surface_tension=nothing,
                                           reference_particle_spacing=fluid_particle_spacing)

# ------------------------------------------------------------------------------
# Boundary System Setup
# ------------------------------------------------------------------------------

boundary_density_calculator = AdamiPressureExtrapolation()
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation, # Use same EoS for boundary pressure
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length,
                                             correction=nothing,
                                             reference_particle_spacing=fluid_particle_spacing)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model,
                                    adhesion_coefficient=0.0) # No adhesion

# ------------------------------------------------------------------------------
# Simulation:
# ------------------------------------------------------------------------------

# Semidiscretization combines all systems and defines neighborhood search strategy.
# `update_strategy=nothing` allows TrixiParticles to choose an optimal update strategy.
semi = Semidiscretization(fluid_system, boundary_system,
                          neighborhood_search=GridNeighborhoodSearch{2}(update_strategy=nothing),
                          parallelization_backend=PolyesterBackend())

# Create OrdinaryDiffEq.jl problem
ode = semidiscretize(semi, tspan)

# Callbacks for monitoring, saving, and adaptive time stepping
info_callback = InfoCallback(interval=250) # Print simulation info every 250 steps

# Regular solution saving
solution_prefix = "" # Prefix for output files
saving_callback = SolutionSavingCallback(dt=0.05, prefix=solution_prefix) # Save every 0.05 time units

# Save solution at specific time points for comparison with Marrone et al. (2011)
# Note: Marrone et al. results are at a much higher resolution (dx = H/320).
# Times correspond to t*sqrt(g/H) = [1.5, 2.36, 3.0, 5.7, 6.45] for H=0.6
# H = initial_water_height
time_factor = sqrt(initial_water_height / gravity)
paper_save_times = [0.0, 1.5, 2.36, 3.0, 5.7, 6.45] .* time_factor
saving_paper_callback = SolutionSavingCallback(save_times=paper_save_times,
                                               prefix="marrone_times")

# Optional callback for density reinitialization (Shepard filter)
use_density_reinitialization = false
density_reinit_callback = use_density_reinitialization ?
                          DensityReinitializationCallback(semi.systems[1], interval=10) :
                          nothing

stepsize_callback = StepsizeCallback(cfl=0.9) # CFL-based adaptive time stepping

# Group all callbacks. `extra_callback` can be passed via `trixi_include`.
extra_callback = nothing
callbacks = CallbackSet(info_callback, saving_callback, stepsize_callback,
                        saving_paper_callback, density_reinit_callback, extra_callback)

# CarpenterKennedy2N54 is a popular choice for SPH.
# `dt=1.0` is an initial guess; `StepsizeCallback` will override it.
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # This is overwritten by the stepsize callback
            save_everystep=false,
            callback=callbacks)

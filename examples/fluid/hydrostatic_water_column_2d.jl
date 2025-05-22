# ==========================================================================================
# 2D Hydrostatic Water Column Simulation
#
# This example simulates a column of water at rest in a tank under gravity.
# It is a basic test case to verify hydrostatic pressure distribution and stability.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------

# Particle spacing, determines the resolution of the simulation
particle_spacing = 0.05

# Number of boundary particle layers.
# Chosen to ensure full kernel support for fluid particles near the boundary.
boundary_layers = 3

# Gravitational acceleration
gravity = 9.81

# Default gravitational acceleration vector for the system.
system_acceleration = (0.0, -gravity)

# Simulation time span
tspan = (0.0, 1.0)

# ------------------------------------------------------------------------------
# Experiment Setup
# ------------------------------------------------------------------------------
# Dimensions of the initial fluid column and the containing tank
initial_fluid_size = (1.0, 0.9) # width, height
tank_size = (1.0, 1.0)          # width, height

# Fluid properties
fluid_density = 1000.0
sound_speed = 10.0

state_equation = StateEquationCole(sound_speed=sound_speed,
                                   reference_density=fluid_density,
                                   exponent=7,
                                   clip_negative_pressure=false)

# Setup for the rectangular tank with fluid and boundary particles.
tank = RectangularTank(particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers,
                       acceleration=system_acceleration,
                       state_equation=state_equation)

# ------------------------------------------------------------------------------
# Fluid System Setup
# ------------------------------------------------------------------------------

smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

fluid_density_calculator = ContinuityDensity()

alpha = 0.02
viscosity = ArtificialViscosityMonaghan(alpha=alpha, beta=0.0)

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length,
                                           viscosity=viscosity,
                                           acceleration=system_acceleration,
                                           source_terms=nothing,
                                           reference_particle_spacing=particle_spacing)

# ------------------------------------------------------------------------------
# Boundary System Setup
# ------------------------------------------------------------------------------

boundary_density_calculator = AdamiPressureExtrapolation()

# This `viscosity_wall` can be overridden by `trixi_include`.
viscosity_wall = nothing

boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation, # Use same EoS for boundary pressure
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length, # Use fluid's kernel and S.L.
                                             viscosity=viscosity_wall,
                                             reference_particle_spacing=particle_spacing)

# `movement=nothing` indicates static boundaries by default. This can be overridden.
boundary_system = BoundarySPHSystem(tank.boundary, boundary_model, movement=nothing)

# ------------------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------------------

semi = Semidiscretization(fluid_system, boundary_system,
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50) # Print simulation info every 50 steps
saving_callback = SolutionSavingCallback(dt=0.05, prefix="") # Save solution every 0.05 time units

# Placeholder for additional callbacks that can be passed via `trixi_include`.
extra_callback = nothing

callbacks = CallbackSet(info_callback, saving_callback, extra_callback)

# Solve the ODE system using a Runge-Kutta method with adaptive time stepping.
sol = solve(ode, RDPK3SpFSAL35(), save_everystep=false, callback=callbacks)

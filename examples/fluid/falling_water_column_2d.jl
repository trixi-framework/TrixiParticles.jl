# ==========================================================================================
# 2D Falling Water Column Simulation
#
# This example simulates a column of water falling under gravity within a closed tank.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ------------------------------------------------------------------------------
# Resolution Parameters
# ------------------------------------------------------------------------------
# Particle spacing, determines the resolution of the simulation
fluid_particle_spacing = 0.02

# For Monaghan-Kajtar boundary model, typical values are spacing_ratio=3 and boundary_layers=1.
boundary_layers = 3 # Number of boundary particle layers
spacing_ratio = 1   # Ratio of fluid particle spacing to boundary particle spacing

# ------------------------------------------------------------------------------
# Experiment Setup
# ------------------------------------------------------------------------------
# Gravitational acceleration
gravity = -9.81
gravity_vec = (0.0, gravity)

# Simulation time span
tspan = (0.0, 2.0)

# Dimensions of the initial water column and the containing tank
initial_fluid_size = (0.5, 1.0) # width, height
tank_size = (4.0, 4.0)          # width, height

# Fluid properties
fluid_density = 1000.0 # Reference density of water (kg/m^3)
sound_speed = 10 * sqrt(abs(gravity) * initial_fluid_size[2])

# Equation of state for the fluid (weakly compressible)
state_equation = StateEquationCole(sound_speed=sound_speed,
                                   reference_density=fluid_density,
                                   exponent=7) # Tait-like equation of state

# Setup for the rectangular tank with fluid and boundary particles
tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers,
                       spacing_ratio=spacing_ratio,
                       acceleration=gravity_vec,
                       state_equation=state_equation)

# Reposition the initial water column within the tank
# Move water column to be centered horizontally and slightly elevated
fluid_offset_x = 0.5 * tank_size[1] - 0.5 * initial_fluid_size[1]
fluid_offset_y = 0.2 # Initial elevation from the tank bottom
fluid_initial_position_offset = SVector(fluid_offset_x, fluid_offset_y)

for i in axes(tank.fluid.coordinates, 2)
    tank.fluid.coordinates[:, i] .+= fluid_initial_position_offset
end

# ------------------------------------------------------------------------------
# Fluid System Setup
# ------------------------------------------------------------------------------

smoothing_length = 1.2 * fluid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

# Density calculation method and viscosity model
fluid_density_calculator = ContinuityDensity()
viscosity_model = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length,
                                           viscosity=viscosity_model,
                                           acceleration=gravity_vec,
                                           reference_particle_spacing=fluid_particle_spacing)

# ------------------------------------------------------------------------------
# Boundary System Setup
# ------------------------------------------------------------------------------

boundary_density_calculator = AdamiPressureExtrapolation()
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length,
                                             reference_particle_spacing=fluid_particle_spacing)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

# ------------------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------------------

# Semidiscretization combines all systems and defines neighborhood search strategy.
semi = Semidiscretization(fluid_system, boundary_system,
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=250) # Print simulation info every 250 steps
saving_callback = SolutionSavingCallback(dt=0.05, prefix="") # Save solution every 0.05 time units

callbacks = CallbackSet(info_callback, saving_callback)

# Solve the ODE system using a Runge-Kutta method with adaptive time stepping.
# `abstol` and `reltol` control the time step adaptivity based on error estimation.
# `dtmax` limits the maximum allowed time step to prevent instabilities,
# especially when particles are close to boundaries.
sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-5, # Absolute tolerance for time integration (default: 1e-6)
            reltol=1e-3, # Relative tolerance for time integration (default: 1e-3)
                         # May need tuning to prevent boundary penetration or excessive damping.
            dtmax=1e-2,  # Maximum allowed time step
            save_everystep=false,
            callback=callbacks)

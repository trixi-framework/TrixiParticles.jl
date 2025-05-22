# ==========================================================================================
# 2D Moving Wall Simulation
#
# This example simulates a column of water in a tank where one of the vertical walls
# moves horizontally, pushing the fluid.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------
# Particle spacing, determines the resolution of the simulation
particle_spacing = 0.05

# Number of boundary particle layers
boundary_layers = 3

# Gravitational acceleration
gravity = 9.81
gravity_vec = (0.0, -gravity)

# Simulation time span
tspan = (0.0, 2.0)

# ------------------------------------------------------------------------------
# Experiment Setup: Tank, Fluid, and Wall Configuration
# ------------------------------------------------------------------------------
# Dimensions of the initial fluid column and the containing tank
initial_fluid_size = (1.0, 0.8) # width, height
tank_size = (4.0, 1.0)          # width, height of the overall domain

# Fluid properties
fluid_density = 1000.0
sound_speed = 10 * sqrt(gravity * initial_fluid_size[2])

state_equation = StateEquationCole(sound_speed=sound_speed,
                                   reference_density=fluid_density,
                                   exponent=7)

# Setup for the rectangular tank with fluid and boundary particles.
# Initially, all walls are static.
tank = RectangularTank(particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers,
                       spacing_ratio=1.0,
                       acceleration=gravity_vec,
                       state_equation=state_equation)

# Modify the tank setup: one wall (right wall, face_indices[2]) will be made movable.
# `reset_wall!` reconfigures specified faces of the tank.
# Here, we re-initialize the right wall. The other walls remain as initially defined.
# `(false, true, false, false)` flags: (left, right, bottom, top)
# The new position/extent for the right wall is set to `tank.fluid_size[1]`,
# effectively placing it at the initial right edge of the fluid.
TrixiParticles.reset_wall!(tank, (false, true, false, false),
                           (0.0, tank.fluid_size[1], 0.0, 0.0))

# ------------------------------------------------------------------------------
# Boundary Movement Setup
# ------------------------------------------------------------------------------

# Define the movement of the right wall.
# S(t) = (0.5*a*t^2, 0.0) where 'a' is some acceleration (here implicitly 1.0).
# The wall moves in the positive x-direction.
movement_function(t) = SVector(0.5 * t^2, 0.0)

# The wall moves only for t < 1.5 seconds.
is_moving_wall(t) = t < 1.5

# Create BoundaryMovement object, specifying which particles are moving.
# `tank.face_indices[2]` refers to the particles of the right wall.
boundary_movement = BoundaryMovement(movement_function, is_moving_wall,
                                     moving_particles=tank.face_indices[2])

# ------------------------------------------------------------------------------
# Fluid System Setup
# ------------------------------------------------------------------------------

smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

fluid_density_calculator = ContinuityDensity()
viscosity_model = ArtificialViscosityMonaghan(alpha=0.1, beta=0.0)

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length,
                                           viscosity=viscosity_model,
                                           acceleration=gravity_vec,
                                           reference_particle_spacing=particle_spacing)

# ------------------------------------------------------------------------------
# Boundary System Setup
# ------------------------------------------------------------------------------

boundary_density_calculator = AdamiPressureExtrapolation()
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length,
                                             reference_particle_spacing=particle_spacing)

# Assign the defined movement to the boundary system.
boundary_system = BoundarySPHSystem(tank.boundary, boundary_model,
                                    movement=boundary_movement)

# ------------------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------------------
semi = Semidiscretization(fluid_system, boundary_system,
                          parallelization_backend=PolyesterBackend())
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")
callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so because forces become extremely large when
# fluid particles are very close to boundary particles, and the time integration method
# interprets this as an instability.
sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false,
            callback=callbacks);

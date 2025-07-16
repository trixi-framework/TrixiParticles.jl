# ==========================================================================================
# 2D Moving Wall Simulation
#
# This example simulates a column of water in a tank where one of the vertical walls
# moves horizontally, pushing the fluid.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.05

# Make sure that the kernel support of fluid particles at a boundary is always fully sampled
boundary_layers = 3

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 2.0)

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (1.0, 0.8)
tank_size = (4.0, 1.0)

fluid_density = 1000.0
sound_speed = 10 * sqrt(gravity * initial_fluid_size[2])
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=1.0,
                       acceleration=(0.0, -gravity), state_equation=state_equation)

reset_wall!(tank, (false, true, false, false), (0.0, tank.fluid_size[1], 0.0, 0.0))

# Movement function
movement_function(t) = SVector(0.5t^2, 0.0)

is_moving(t) = t < 1.5

boundary_movement = BoundaryMovement(movement_function, is_moving,
                                     moving_particles=tank.face_indices[2])

# ==========================================================================================
# ==== Fluid
smoothing_length = 1.2 * fluid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

fluid_density_calculator = ContinuityDensity()
viscosity = ArtificialViscosityMonaghan(alpha=0.1, beta=0.0)

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length, viscosity=viscosity,
                                           acceleration=(0.0, -gravity))

# ==========================================================================================
# ==== Boundary
boundary_density_calculator = AdamiPressureExtrapolation()
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model,
                                    movement=boundary_movement)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, boundary_system)
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
            save_everystep=false, callback=callbacks);

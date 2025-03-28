using TrixiParticles
using OrdinaryDiffEq

gravity = -9.81

# ==========================================================================================
# ==== Falling Rocks Simulation Setup
#
# This example sets up a simulation of falling rocks within a rectangular tank.
# The tank is generated using a helper function (RectangularTank) which creates
# a "fluid" region (containing the rock particles) and a "boundary" region (representing walls).
# ==========================================================================================

particle_spacing = 0.1

rock_width = 2.0
rock_height = 2.0
rock_density = 3000.0

tank_width = 2.0
tank_height = 4.0

# Create a rectangular tank. The tank has a "fluid" region for the rock particles
# and a "boundary" region for the container walls.
tank = RectangularTank(particle_spacing, (rock_width, rock_height),
                       (tank_width, tank_height), rock_density,
                       n_layers=2)

# ==========================================================================================
# ==== Systems Setup
#
# We adjust the rock positions upward to allow them to fall under gravity.
# Next, we create a contact model and use it to build the rock (DEM) system.
# The contact model is dispatched using multiple types (e.g. Hertz or Linear).
# ==========================================================================================

# Move the rock particles up to let them fall
tank.fluid.coordinates[2, :] .+= 0.5

# Create a contact model.
# Option 1: Hertzian contact model (uses elastic modulus and Poisson's ratio)
contact_model = HertzContactModel(10e9, 0.3)
# Option 2 (alternative): Linear contact model (constant normal stiffness)
# contact_model = LinearContactModel(2 * 10e5)

# Construct the rock system using the new DEMSystem signature.
rock_system = DEMSystem(tank.fluid, contact_model; damping_coefficient=0.0001,
                        acceleration=(0.0, gravity))

# Construct the boundary system for the tank walls.
boundary_system = BoundaryDEMSystem(tank.boundary, 10e7)

# ==========================================================================================
# ==== Simulation Setup
#
# We now create a semidiscretization (spatially discretized system) and set up
# the time integration using a Runge-Kutta method with adaptive time-stepping.
# ==========================================================================================

semi = Semidiscretization(rock_system, boundary_system)

tspan = (0.0, 4.0)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=5000)
saving_callback = SolutionSavingCallback(dt=0.02)
callbacks = CallbackSet(info_callback, saving_callback)

# Solve the ODE using a Runge-Kutta method with adaptive (error-based) time step control.
sol = solve(ode, RDPK3SpFSAL49();
            abstol=1e-5,   # Absolute tolerance (tuning may be necessary to avoid boundary penetration)
            reltol=1e-4,   # Relative tolerance (tuning may be necessary)
            dtmax=1e-3,    # Maximum time step (limits large steps that may destabilize the simulation)
            dt=1e-7,       # Initial time step
            save_everystep=false, callback=callbacks)

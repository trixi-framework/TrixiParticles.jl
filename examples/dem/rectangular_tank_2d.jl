# ==========================================================================================
# ==== Falling Rocks Simulation Setup
#
# Two tanks are created with identical geometric and material parameters.
# Their rock particles are raised slightly to let them fall under gravity.
# Tank 1 uses a Hertzian contact model; Tank 2 uses a Linear contact model.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

gravity = -9.81

particle_spacing = 0.1

rock_width = 2.0
rock_height = 2.0
rock_density = 3000.0

tank_width = 2.0
tank_height = 4.0

# Create two rectangular tanks.
tank1 = RectangularTank(particle_spacing, (rock_width, rock_height),
                        (tank_width, tank_height), rock_density,
                        n_layers=2)
tank2 = RectangularTank(particle_spacing, (rock_width, rock_height),
                        (tank_width, tank_height), rock_density,
                        n_layers=2, min_coordinates=(tank_width + 0.5, 0.0))

# ==========================================================================================
# ==== Systems
# ==== Adjust Particle Positions
#
# We adjust the rock positions upward to allow them to fall under gravity.
# Next, we create a contact model and use it to build the rock (DEM) system.
# Raise the rock particles so that they fall under gravity.
# Shift tank2 horizontally to avoid overlap.
# ==========================================================================================

# Move the rock particles up to let them fall
# Move the rock particles up to let them fall
tank.fluid.coordinates[2, :] .+= 0.5
# small perturbation
tank.fluid.coordinates .+= 0.01 .* (2 .* rand(size(tank.fluid.coordinates)) .- 1)
tank1.fluid.coordinates[2, :] .+= 0.5

# Create a contact model.
# Option 1: Hertzian contact model (uses elastic modulus and Poisson's ratio)
contact_model = HertzContactModel(10e9, 0.3)
# Option 2 (alternative): Linear contact model (constant normal stiffness)
# contact_model = LinearContactModel(2 * 10e5)

# Construct the rock system using the new DEMSystem signature.
rock_system = DEMSystem(tank.fluid, contact_model; damping_coefficient=0.0001,
                        acceleration=(0.0, gravity), radius=0.4 * particle_spacing)

# Construct the boundary system for the tank walls.
boundary_system = BoundaryDEMSystem(tank.boundary, 10e7)
tank2.fluid.coordinates[2, :] .+= 0.5

# ==========================================================================================
# ==== Simulation
# ==========================================================================================
# Contact model for tank1: Hertzian contact model.
contact_model1 = HertzContactModel(10e9, 0.3)
# Contact model for tank2: Linear contact model.
contact_model2 = LinearContactModel(6 * 10e5)

# Construct DEM systems for each tank.
rock_system1 = DEMSystem(tank1.fluid, contact_model1;
                         damping_coefficient=0.0001,
                         acceleration=(0.0, gravity))
rock_system2 = DEMSystem(tank2.fluid, contact_model2;
                         damping_coefficient=0.0001,
                         acceleration=(0.0, gravity))

# Construct the boundary systems for the tank walls.
boundary_system1 = BoundaryDEMSystem(tank1.boundary, 10e7)
boundary_system2 = BoundaryDEMSystem(tank2.boundary, 10e7)

semi = Semidiscretization(rock_system1, rock_system2,
                          boundary_system1, boundary_system2)

tspan = (0.0, 4.0)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=5000)
saving_callback = SolutionSavingCallback(dt=0.02)
callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-3, # Limit stepsize to prevent crashing
            dt=1e-7,  # Initial step size
            save_everystep=false, callback=callbacks);

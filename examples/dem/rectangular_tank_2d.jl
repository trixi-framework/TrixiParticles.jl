using TrixiParticles
using OrdinaryDiffEq

gravity = -9.81

# ==========================================================================================
# ==== Falling Rocks Simulation Setup
#
# Two tanks are created with identical geometric and material parameters.
# Their rock particles are raised slightly to let them fall under gravity.
# Tank 1 uses a Hertzian contact model; Tank 2 uses a Linear contact model.
# ==========================================================================================

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
# ==== Adjust Particle Positions
#
# Raise the rock particles so that they fall under gravity.
# Shift tank2 horizontally to avoid overlap.
# ==========================================================================================
tank1.fluid.coordinates[2, :] .+= 0.5

tank2.fluid.coordinates[2, :] .+= 0.5

# ==========================================================================================
# ==== Systems Setup
#
# Create contact models and build DEM systems.
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

sol = solve(ode, RDPK3SpFSAL49();
            abstol=1e-5,
            reltol=1e-4,
            dtmax=1e-3,
            dt=1e-7,
            save_everystep=false, callback=callbacks)

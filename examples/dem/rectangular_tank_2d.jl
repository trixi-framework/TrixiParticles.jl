using TrixiParticles
using OrdinaryDiffEq

gravity = -9.81

# ==========================================================================================
# ==== Fluid

particle_spacing = 0.1

# Ratio of fluid particle spacing to boundary particle spacing
beta = 1
boundary_layers = 2

rock_width = 2.0
rock_height = 2.0
rock_density = 3000.0

tank_width = 2.0
tank_height = 4.0

tank = RectangularTank(particle_spacing, (rock_width, rock_height),
                       (tank_width, tank_height), rock_density,
                       n_layers=boundary_layers, spacing_ratio=beta)

# ==========================================================================================
# ==== Systems

# let them fall
tank.fluid.coordinates[2, :] .+= 0.5
solid_system = DEMSystem(tank.fluid, 2 * 10e5, 10e9, 0.3, acceleration=(0.0, gravity))
boundary_system = BoundaryDEMSystem(tank.boundary, 10e7)

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(solid_system, boundary_system,
                          neighborhood_search=GridNeighborhoodSearch)

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
saving_callback = SolutionSavingCallback(dt=0.02)

callbacks = CallbackSet(info_callback, saving_callback)

sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-3, # Limit stepsize to prevent crashing
            dt=1e-7,
            save_everystep=false, callback=callbacks);

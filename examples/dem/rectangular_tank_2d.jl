using TrixiParticles
using OrdinaryDiffEq

gravity = -9.81

# ==========================================================================================
# ==== Falling rocks

particle_spacing = 0.1

rock_width = 2.0
rock_height = 2.0
rock_density = 3000.0

tank_width = 2.0
tank_height = 4.0

tank = RectangularTank(particle_spacing, (rock_width, rock_height),
                       (tank_width, tank_height), rock_density,
                       n_layers=2)

# ==========================================================================================
# ==== Systems

# Move the rocks up to let them fall
tank.fluid.coordinates[2, :] .+= 0.5
rock_system = DEMSystem(tank.fluid, 2 * 10e5, 10e9, 0.3, acceleration=(0.0, gravity))
boundary_system = BoundaryDEMSystem(tank.boundary, 10e7)

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(rock_system, boundary_system,
                          neighborhood_search=GridNeighborhoodSearch)

tspan = (0.0, 5.0)
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

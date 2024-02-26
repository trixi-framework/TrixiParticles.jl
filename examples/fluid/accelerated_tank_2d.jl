# This setup is identical to `hydrostatic_water_column_2d.jl`, except that now there is
# no gravity, and the tank is accelerated upwards instead.
# Note that the two setups are physically identical, but produce different numerical errors.
using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.05

# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
boundary_layers = 3

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 1.0)

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (1.0, 0.9)
tank_size = (1.0, 1.0)

fluid_density = 1000.0
sound_speed = 10.0

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=1.0)

# Function for moving boundaries
movement_function(t) = SVector(0.0, 0.5 * 9.81 * t^2)

is_moving(t) = true

boundary_movement = BoundaryMovement(movement_function, is_moving)

# Import the setup from `hydrostatic_water_column_2d.jl`
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"),
              fluid_particle_spacing=fluid_particle_spacing, movement=boundary_movement,
              acceleration=(0.0, 0.0), tank=tank, semi=nothing, ode=nothing,
              sol=nothing) # Overwrite `sol` assignment to skip time integration

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, boundary_system)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
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
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-3, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);

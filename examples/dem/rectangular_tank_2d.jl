# ==========================================================================================
# ==== Falling Rocks Simulation Setup
#
# This example sets up a simulation of falling rocks within a rectangular tank.
# The tank is generated using a helper function (RectangularTank) which creates
# a "fluid" region (containing the rock particles) and a "boundary" region (representing walls).
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

# Create a rectangular tank. The tank has a "fluid" region for the rock particles
# and a "boundary" region for the container walls.
tank = RectangularTank(particle_spacing, (rock_width, rock_height),
                       (tank_width, tank_height), rock_density,
                       n_layers=2, coordinates_eltype=Float64)

# ==========================================================================================
# ==== Systems
#
# We adjust the rock positions upward to allow them to fall under gravity.
# Next, we create a contact model and use it to build the rock (DEM) system.
# ==========================================================================================

# Move the rock particles up to let them fall
tank.fluid.coordinates[2, :] .+= 0.5
# Small perturbation
for i in 1:size(tank.fluid.coordinates, 2)
    tank.fluid.coordinates[:, i] .+= 0.01 .* (2 .* rand(2) .- 1)
end

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

# ==========================================================================================
# ==== Simulation
# ==========================================================================================
semi = Semidiscretization(rock_system, boundary_system;
                          neighborhood_search=GridNeighborhoodSearch{2}(),
                          parallelization_backend=PolyesterBackend())

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

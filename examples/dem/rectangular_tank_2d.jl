# ==========================================================================================
# 2D Falling Rocks in a Rectangular Tank (DEM)
#
# This example simulates a collection of "rocks" (represented as DEM particles)
# falling under gravity within a 2D rectangular tank.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ------------------------------------------------------------------------------
# Physical Parameters
# ------------------------------------------------------------------------------
# Gravitational acceleration (acting in the negative y-direction)
gravity = 9.81
system_acceleration = (0.0, -gravity)

# Rock particle properties
rock_particle_density = 3000.0 # kg/m^3

# Contact model parameters for rock-rock and rock-boundary interaction
# Option 1: Hertzian Contact Model (more physically based for elastic spheres)
hertz_elastic_modulus = 1.0e10 # Pa (Young's modulus)
hertz_poissons_ratio = 0.3    # Poisson's ratio
# contact_model_rocks = HertzContactModel(hertz_elastic_modulus, hertz_poissons_ratio)

# Option 2: Linear Contact Model (simpler, constant normal stiffness) - Active by default
linear_contact_stiffness_rocks = 2.0e6 # Normal stiffness
contact_model_rocks = LinearContactModel(linear_contact_stiffness_rocks)

# Damping coefficient for inter-particle and particle-boundary collisions
dem_damping_coefficient = 1.0e-4

# Boundary (tank wall) properties
boundary_contact_stiffness = 1.0e8 # Normal stiffness for particle-wall interaction

# ------------------------------------------------------------------------------
# Simulation Geometry and Resolution
# ------------------------------------------------------------------------------
# Particle spacing (characteristic size of rock particles)
particle_spacing = 0.1 # meters

# Initial block of rocks: dimensions
initial_rock_block_width = 2.0  # meters
initial_rock_block_height = 2.0 # meters
initial_rock_block_size = (initial_rock_block_width, initial_rock_block_height)

# Tank dimensions (width, height)
tank_width = 2.0  # meters
tank_height = 4.0 # meters
tank_domain_size = (tank_width, tank_height)

# Number of boundary particle layers for the tank walls
num_boundary_particle_layers_tank = 2

# ------------------------------------------------------------------------------
# Particle Setup: Rocks and Tank Walls
# ------------------------------------------------------------------------------
# Create a RectangularTank object. This helper function generates two sets of particles:
# 1. `tank.fluid`: Intended for fluid, but here used for the initial block of rock particles.
# 2. `tank.boundary`: Particles representing the tank walls.
tank_setup = RectangularTank(particle_spacing,
                             initial_rock_block_size, # Size of the "fluid" region (rocks)
                             tank_domain_size,        # Overall size of the tank
                             rock_particle_density,   # Density for the rock particles
                             n_layers=num_boundary_particle_layers_tank)

# Adjust initial position of rock particles:
# Move the block of rocks upwards to allow them to fall.
vertical_shift_rocks = 0.5 # meters
tank_setup.fluid.coordinates[2, :] .+= vertical_shift_rocks

# Add a small random perturbation to initial rock particle positions
# to break perfect symmetries and promote more natural packing/settling.
perturbation_magnitude = 0.01 * particle_spacing
tank_setup.fluid.coordinates .+= perturbation_magnitude .*
                                 (2 .* rand(size(tank_setup.fluid.coordinates)) .- 1)

# Extract rock particles and boundary particles from the tank setup
rock_particles = tank_setup.fluid
tank_wall_particles = tank_setup.boundary

# ------------------------------------------------------------------------------
# DEM System Setup
# ------------------------------------------------------------------------------
# Effective radius for DEM collision detection.
dem_particle_radius = 0.4 * particle_spacing

# DEM system for the rock particles
rock_dem_system = DEMSystem(rock_particles, contact_model_rocks;
                            damping_coefficient=dem_damping_coefficient,
                            acceleration=system_acceleration,
                            radius=dem_particle_radius)

# Boundary DEM system for the tank walls
tank_boundary_dem_system = BoundaryDEMSystem(tank_wall_particles,
                                             boundary_contact_stiffness)

# ------------------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------------------
# Combine DEM systems into a semidiscretization
semi = Semidiscretization(rock_dem_system, tank_boundary_dem_system,
                          parallelization_backend=PolyesterBackend()) # Default neighborhood search

# Simulation time span
tspan = (0.0, 4.0) # seconds

ode = semidiscretize(semi, tspan)

# Callbacks for monitoring and saving results
info_callback = InfoCallback(interval=5000) # Print info every 5000 steps
saving_callback = SolutionSavingCallback(dt=0.02, prefix="") # Save every 0.02s
callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-3, # Limit stepsize to prevent crashing
            dt=1e-7,  # Initial step size
            save_everystep=false, callback=callbacks);

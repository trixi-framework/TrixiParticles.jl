using TrixiParticles
using OrdinaryDiffEq

gravity = -9.81
acceleration = (0.0, 0.0, gravity)

# ==========================================================================================
# ==== Sandpile Simulation Setup
# ==========================================================================================

# Set particle spacing
particle_spacing = 0.1 # Smaller spacing for a finer resolution sandpile

# Dimensions and Center of the initial sand column
pile_width = 0.5  # x-dimension
pile_depth = 0.5  # y-dimension
pile_height = 1.0 # z-dimension

# Place bottom of pile slightly above z=0 to avoid initial boundary layer penetration
pile_min_z = particle_spacing * 0.1 # Start above z=0
pile_center_z = pile_min_z + pile_height / 2
pile_center = (0.0, 0.0, pile_center_z)

# Density of sand particles
sand_density = 1600.0 # kg/m^3 (typical for loose dry sand)

# Dimensions of the bounding box (container)
container_width = 10  # x-dimension
container_depth = 10  # y-dimension
container_height = 1.5 # z-dimension

# Number of layers for the boundary particles (used for thickness calc)
n_boundary_layers = 3
boundary_thickness = n_boundary_layers * particle_spacing # Assumed thickness for placement

# ==========================================================================================
# ==== Particle Generation using RectangularShape and union
# ==========================================================================================

# --- Sand Column Particles ---
# Calculate number of particles and min coordinates for the sand column
n_particles_pile_x = round(Int, pile_width / particle_spacing)
n_particles_pile_y = round(Int, pile_depth / particle_spacing)
n_particles_pile_z = round(Int, pile_height / particle_spacing)
n_particles_pile = (n_particles_pile_x, n_particles_pile_y, n_particles_pile_z)

min_coords_pile = (pile_center[1] - pile_width / 2,
                   pile_center[2] - pile_depth / 2,
                   pile_min_z) # Use pile_min_z for bottom

# Create the sand particles using RectangularShape
sand_particles = RectangularShape(particle_spacing, n_particles_pile, min_coords_pile;
                                  density=sand_density, coordinates_perturbation=0.1)

# --- Boundary Particles (Floor and Walls) ---
# Define boundary dimensions based on container size
min_boundary = (-container_width / 2, -container_depth / 2, 0.0)
max_boundary = (container_width / 2, container_depth / 2, container_height)

# Use tlsph=true for boundaries to place particles *at* the boundary edge
# The density for boundary particles is often just a placeholder if mass isn't used.
boundary_density = sand_density # Or 1000.0, or whatever BoundaryDEMSystem assumes

# Floor
# Extend floor slightly beyond the main container dimensions to ensure wall bases sit fully on it
floor_width = container_width + 2 * boundary_thickness
floor_depth = container_depth + 2 * boundary_thickness
n_particles_floor_x = round(Int, floor_width / particle_spacing)
n_particles_floor_y = round(Int, floor_depth / particle_spacing)
n_particles_floor_z = n_boundary_layers
min_coords_floor = (min_boundary[1] - boundary_thickness,
                    min_boundary[2] - boundary_thickness,
                    min_boundary[3] - boundary_thickness) # Start layers below z=0
floor_particles = RectangularShape(particle_spacing,
                                   (n_particles_floor_x, n_particles_floor_y,
                                    n_particles_floor_z),
                                   min_coords_floor; density=boundary_density, tlsph=true)

boundary_particles = floor_particles

# ==========================================================================================
# ==== Systems Setup
# ==========================================================================================

# contact_model = HertzContactModel(1.0e7, 0.3)
contact_model = LinearContactModel(1e6)

damping_coefficient = 0.00001

# Construct the sand particle system (DEMSystem)
sand_system = DEMSystem(sand_particles, contact_model;
                        damping_coefficient=damping_coefficient,
                        acceleration=acceleration, radius=0.4 * particle_spacing)

# Construct the boundary system for the container walls.
# Boundary stiffness needs to be high enough to prevent excessive penetration.
boundary_stiffness = 1.0e5
boundary_system = BoundaryDEMSystem(boundary_particles, boundary_stiffness)

# ==========================================================================================
# ==== Simulation Setup
# ==========================================================================================

# Combine systems into a semidiscretization
semi = Semidiscretization(sand_system, boundary_system)

# Define the simulation time span
tspan = (0.0, 2.0)

# Create the ODE problem
ode = semidiscretize(semi, tspan)

# Set up callbacks for monitoring and saving results
info_callback = InfoCallback(interval=100) # Print info every 100 steps
saving_callback = SolutionSavingCallback(dt=0.01, # Save every 0.01 simulation seconds
                                         prefix="")
callbacks = CallbackSet(info_callback, saving_callback)

# Choose an ODE solver
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-6,     # Absolute tolerance
            reltol=1e-5,     # Relative tolerance
            dtmax=1e-3,      # Maximum allowed time step
            dt=1e-7,         # Initial time step
            save_everystep=false, # Only save based on saving_callback
            callback=callbacks);

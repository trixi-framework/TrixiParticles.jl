# ==========================================================================================
# 3D Collapsing Sand Pile Simulation (DEM)
#
# This example simulates the collapse of a cylindrical pile of sand under gravity
# within a confined container using the Discrete Element Method (DEM).
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ------------------------------------------------------------------------------
# Physical Parameters
# ------------------------------------------------------------------------------
# Gravitational acceleration (acting in the negative z-direction)
gravity = 9.81
system_acceleration = (0.0, 0.0, -gravity)

# Sand properties
sand_particle_density = 1600.0 # kg/m^3

# Contact model parameters for sand-sand and sand-boundary interaction
sand_contact_stiffness = 1.0e6 # Normal stiffness for LinearContactModel
sand_damping_coefficient = 1.0e-4 # Damping coefficient for inter-particle and particle-boundary collisions

# Boundary (container) properties
boundary_contact_stiffness = 1.0e5 # Normal stiffness for particle-boundary interaction

# ------------------------------------------------------------------------------
# Simulation Geometry and Resolution
# ------------------------------------------------------------------------------
# Particle spacing (characteristic size of sand grains)
particle_spacing = 0.1 # meters

# Initial sand pile (column) dimensions and placement
initial_pile_diameter_xy = 0.5 # meters
initial_pile_height_z = 1.0  # meters
initial_pile_base_size = (initial_pile_diameter_xy, initial_pile_diameter_xy) # x, y dimensions
initial_pile_full_size = (initial_pile_base_size..., initial_pile_height_z)

# Position the base of the pile slightly above z=0
pile_min_z_coordinate = particle_spacing * 0.1
pile_center_z_coordinate = pile_min_z_coordinate + initial_pile_height_z / 2
# Center the pile horizontally at (0,0)
pile_center_position = (0.0, 0.0, pile_center_z_coordinate)

# Container dimensions (width, depth, height)
container_width_xy = 10.0 # meters
container_depth_xy = 10.0 # meters
container_height_z = 1.5  # meters

# Boundary (wall/floor) thickness in terms of particle layers
num_boundary_particle_layers = 1
boundary_layer_thickness = num_boundary_particle_layers * particle_spacing

# ------------------------------------------------------------------------------
# Particle Setup: Sand Pile
# ------------------------------------------------------------------------------
# Calculate number of particles for the initial sand pile
num_particles_pile_dims = round.(Int, initial_pile_full_size ./ particle_spacing)

# Define the minimum coordinates for the rectangular shape representing the sand pile
min_coords_pile_x = pile_center_position[1] - initial_pile_base_size[1] / 2
min_coords_pile_y = pile_center_position[2] - initial_pile_base_size[2] / 2
min_coords_pile = (min_coords_pile_x, min_coords_pile_y, pile_min_z_coordinate)

# Create sand particles. A small coordinate perturbation can help avoid perfect packing.
sand_particles = RectangularShape(particle_spacing, num_particles_pile_dims,
                                  min_coords_pile;
                                  density=sand_particle_density,
                                  coordinates_perturbation=0.1)

# ------------------------------------------------------------------------------
# Particle Setup: Container Boundary (Floor and Walls)
# ------------------------------------------------------------------------------
# Define the inner minimum coordinates of the container volume
min_container_boundary_coords = (-container_width_xy / 2,
                                 -container_depth_xy / 2,
                                 0.0) # Floor at z=0

# Density for boundary particles (can be same as sand or different)
boundary_particle_density = sand_particle_density

# Floor dimensions including boundary layers extending outwards
floor_particles_width = container_width_xy + 2 * boundary_layer_thickness
floor_particles_depth = container_depth_xy + 2 * boundary_layer_thickness
num_particles_floor_x = round(Int, floor_particles_width / particle_spacing)
num_particles_floor_y = round(Int, floor_particles_depth / particle_spacing)
num_particles_floor_z = num_boundary_particle_layers # Thickness of the floor

# Minimum coordinates for the floor particle block
min_coords_floor_x = min_container_boundary_coords[1] - boundary_layer_thickness
min_coords_floor_y = min_container_boundary_coords[2] - boundary_layer_thickness
min_coords_floor_z = min_container_boundary_coords[3] - boundary_layer_thickness
min_coords_floor_particles = (min_coords_floor_x, min_coords_floor_y, min_coords_floor_z)

# Create floor particles.
# For DEM, boundary particles are typically fixed (not part of a `DEMSystem`).
floor_boundary_particles = RectangularShape(particle_spacing,
                                            (num_particles_floor_x,
                                             num_particles_floor_y,
                                             num_particles_floor_z),
                                            min_coords_floor_particles;
                                            density=boundary_particle_density,
                                            tlsph=true)

container_boundary_particles = floor_boundary_particles

# ------------------------------------------------------------------------------
# DEM System Setup
# ------------------------------------------------------------------------------
# Contact model for sand particle interactions
sand_contact_model = LinearContactModel(sand_contact_stiffness)

# DEM system for the sand particles
# Particle radius for DEM is often slightly less than half the spacing to allow for some "rattling"
# or to represent actual grain size if spacing is just a discretization parameter.
dem_particle_radius = 0.4 * particle_spacing # Effective radius for collision detection
sand_dem_system = DEMSystem(sand_particles, sand_contact_model;
                            damping_coefficient=sand_damping_coefficient,
                            acceleration=system_acceleration,
                            radius=dem_particle_radius)

# Boundary DEM system for the container
# This system defines how DEM particles interact with fixed boundary particles.
container_boundary_dem_system = BoundaryDEMSystem(container_boundary_particles,
                                                  boundary_contact_stiffness)

# ------------------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------------------
# Combine DEM systems into a semidiscretization
semi = Semidiscretization(sand_dem_system, container_boundary_dem_system,
                          parallelization_backend=PolyesterBackend())

# Simulation time span
tspan = (0.0, 2.0) # seconds

ode = semidiscretize(semi, tspan)

# Callbacks for monitoring and saving results
info_callback = InfoCallback(interval=2000) # Print info every 2000 steps
saving_callback = SolutionSavingCallback(dt=0.02, prefix="") # Save every 0.02s
callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-3, # Limit stepsize to prevent crashing
            dt=1e-7,  # Initial step size
            save_everystep=false, callback=callbacks);

using TrixiParticles
using OrdinaryDiffEq

# Physical parameters
gravity = -9.81
acceleration = (0.0, 0.0, gravity)

# -----------------------------------------------------------------------------
# Sandpile Simulation Setup
# -----------------------------------------------------------------------------
particle_spacing = 0.1

# Initial sand column dimensions and placement
pile_width = 0.5
pile_depth = 0.5
pile_height = 1.0
pile_min_z = particle_spacing * 0.1
pile_center_z = pile_min_z + pile_height / 2
pile_center = (0.0, 0.0, pile_center_z)
sand_density = 1600.0

# Container dimensions
container_width = 10
container_depth = 10
container_height = 1.5

n_boundary_layers = 1
boundary_thickness = n_boundary_layers * particle_spacing

# -----------------------------------------------------------------------------
# Particle Generation
# -----------------------------------------------------------------------------
# Sand column particles
n_particles_pile_x = round(Int, pile_width / particle_spacing)
n_particles_pile_y = round(Int, pile_depth / particle_spacing)
n_particles_pile_z = round(Int, pile_height / particle_spacing)
n_particles_pile = (n_particles_pile_x, n_particles_pile_y, n_particles_pile_z)

min_coords_pile = (pile_center[1] - pile_width / 2,
                   pile_center[2] - pile_depth / 2,
                   pile_min_z)
sand_particles = RectangularShape(particle_spacing, n_particles_pile, min_coords_pile;
                                  density=sand_density, coordinates_perturbation=0.1)

# Boundary particles (floor and walls)
min_boundary = (-container_width / 2, -container_depth / 2, 0.0)
boundary_density = sand_density

floor_width = container_width + 2 * boundary_thickness
floor_depth = container_depth + 2 * boundary_thickness
n_particles_floor_x = round(Int, floor_width / particle_spacing)
n_particles_floor_y = round(Int, floor_depth / particle_spacing)
n_particles_floor_z = n_boundary_layers

min_coords_floor = (min_boundary[1] - boundary_thickness,
                    min_boundary[2] - boundary_thickness,
                    min_boundary[3] - boundary_thickness)
floor_particles = RectangularShape(particle_spacing,
                                   (n_particles_floor_x, n_particles_floor_y,
                                    n_particles_floor_z),
                                   min_coords_floor; density=boundary_density, tlsph=true)
boundary_particles = floor_particles

# -----------------------------------------------------------------------------
# Systems Setup
# -----------------------------------------------------------------------------
contact_model = LinearContactModel(1e6)
damping_coefficient = 0.00001

sand_system = DEMSystem(sand_particles, contact_model;
                        damping_coefficient=damping_coefficient,
                        acceleration=acceleration, radius=0.4 * particle_spacing)

boundary_stiffness = 1.0e5
boundary_system = BoundaryDEMSystem(boundary_particles, boundary_stiffness)

# -----------------------------------------------------------------------------
# Simulation Setup
# -----------------------------------------------------------------------------
semi = Semidiscretization(sand_system, boundary_system)
tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.01, prefix="")
callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-3, # Limit stepsize to prevent crashing
            dt=1e-7,  # Initial step size
            save_everystep=false, callback=callbacks);

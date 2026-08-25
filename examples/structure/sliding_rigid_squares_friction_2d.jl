# ==========================================================================================
# 2D Sliding Rigid Squares with and without Wall Friction
#
# Two identical rigid squares slide on the same floor. The left square uses normal-only rigid
# contact, while the right square also uses Coulomb friction. The frictional square slows down
# and starts rotating due to tangential wall forces, whereas the normal-only square keeps
# sliding without spin-up.
#
# In ParaView, compare the trajectories and the rigid-body field data such as
# `angular_velocity`, `contact_count`, and `max_contact_penetration`.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEqLowStorageRK

# ==========================================================================================
# ==== Resolution
particle_spacing = 0.03
wall_particle_spacing = particle_spacing
boundary_layers = 3
contact_distance = 2.0 * particle_spacing

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 0.8)

square_side_length = 0.18
square_density = 1000.0
square_particles_per_side = round(Int, square_side_length / particle_spacing)
# Place the lowest square particles one contact distance above the top wall particles.
square_bottom_y = contact_distance - (particle_spacing + wall_particle_spacing) / 2

square_frictionless = RectangularShape(particle_spacing,
                                       (square_particles_per_side,
                                        square_particles_per_side),
                                       (-1.0, square_bottom_y),
                                       density=square_density,
                                       velocity=(1.0, 0.0))
square_frictional = RectangularShape(particle_spacing,
                                     (square_particles_per_side,
                                      square_particles_per_side),
                                     (0.55, square_bottom_y),
                                     density=square_density,
                                     velocity=(1.0, 0.0))

# ==========================================================================================
# ==== Wall Boundary
floor_length = 3.0
floor_height = 0.03
wall_density = 1000.0

# `RectangularTank` supplies flat-face normals, so rigid-wall contact uses the projected
# distance to the floor and does not require finer wall sampling than the rigid bodies.
floor = RectangularTank(wall_particle_spacing, (0.0, 0.0),
                        (floor_length, floor_height),
                        wall_density, n_layers=boundary_layers,
                        min_coordinates=(-1.5, 0.0),
                        faces=(false, false, true, false))

boundary_model = BoundaryModelMonaghanKajtar(10.0, 1.0, wall_particle_spacing,
                                             floor.boundary.mass)
boundary_system = WallBoundarySystem(floor.boundary, boundary_model)

# ==========================================================================================
# ==== Rigid Structures
contact_model_frictionless = RigidContactModel(; normal_stiffness=2.0e5,
                                               normal_damping=100.0,
                                               contact_distance)

contact_model_frictional = RigidContactModel(; normal_stiffness=2.0e5,
                                             normal_damping=100.0,
                                             static_friction_coefficient=0.6,
                                             kinetic_friction_coefficient=0.4,
                                             tangential_stiffness=1.0e5,
                                             tangential_damping=150.0,
                                             contact_distance)

structure_system_frictionless = RigidBodySystem(square_frictionless;
                                                contact_model=contact_model_frictionless,
                                                acceleration=(0.0, -gravity),
                                                particle_spacing=particle_spacing,
                                                color_value=1)
structure_system_frictional = RigidBodySystem(square_frictional;
                                              contact_model=contact_model_frictional,
                                              acceleration=(0.0, -gravity),
                                              particle_spacing=particle_spacing,
                                              color_value=2)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(structure_system_frictionless, structure_system_frictional,
                          boundary_system)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
saving_callback = SolutionSavingCallback(dt=0.02, output_directory="out", prefix="")

stepsize_callback = StepsizeCallback(cfl=0.5)
callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback(),
                        stepsize_callback)

# Error control handles the free motion while the stepsize callback resolves contact.
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-6,
            reltol=1e-4,
            save_everystep=false, callback=callbacks);

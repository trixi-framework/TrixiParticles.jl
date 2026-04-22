# ==========================================================================================
# 2D Colliding Rigid Spheres
#
# Two rigid disks collide without fluid or walls. Their centers are vertically offset so the
# impact is oblique and generates rotation during the rebound.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEqLowStorageRK

# ==========================================================================================
# ==== Resolution
particle_spacing = 0.03

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 0.8)

sphere_radius = 0.12
sphere_1_center = (-0.25, -0.03)
sphere_2_center = (0.25, 0.03)
sphere_1_density = 1000.0
sphere_2_density = 1200.0

sphere_1_velocity = (0.6, 0.0)
sphere_2_velocity = (-0.4, 0.0)

sphere_1 = SphereShape(particle_spacing, sphere_radius, sphere_1_center, sphere_1_density,
                       sphere_type=RoundSphere(), velocity=sphere_1_velocity)
sphere_2 = SphereShape(particle_spacing, sphere_radius, sphere_2_center, sphere_2_density,
                       sphere_type=RoundSphere(), velocity=sphere_2_velocity)

# ==========================================================================================
# ==== Rigid Structures
contact_model = RigidContactModel(; normal_stiffness=2.0e4,
                                  normal_damping=120.0,
                                  contact_distance=2.0 * particle_spacing)

structure_system_1 = RigidBodySystem(sphere_1;
                                     contact_model=contact_model,
                                     acceleration=(0.0, 0.0),
                                     particle_spacing=particle_spacing)
structure_system_2 = RigidBodySystem(sphere_2;
                                     contact_model=contact_model,
                                     acceleration=(0.0, 0.0),
                                     particle_spacing=particle_spacing)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(structure_system_1, structure_system_2)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=20)
saving_callback = SolutionSavingCallback(dt=0.02, output_directory="out", prefix="")

callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-6,
            reltol=1e-4,
            save_everystep=false, callback=callbacks);

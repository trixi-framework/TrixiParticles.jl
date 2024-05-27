using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.008

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
nu = 0.01
fluid_density = 1000.0
sound_speed = 50

sphere1_radius = 0.05

sphere1_center = (0.5, 0.5, 0.2)
sphere2_center = (1.5, 0.5, 0.2)
sphere1 = SphereShape(fluid_particle_spacing, sphere1_radius, sphere1_center,
                      fluid_density, sphere_type=VoxelSphere(), velocity=(0.0, 0.0, -2.0))
sphere2 = SphereShape(fluid_particle_spacing, sphere1_radius, sphere2_center,
                      fluid_density, sphere_type=VoxelSphere(), velocity=(0.0, 0.0, -2.0))

# `compact_support` needs to be 2.0 * particle_spacing to be correct
fluid_smoothing_length = 2.0 * fluid_particle_spacing

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "falling_water_spheres_2d.jl"),
              fluid_particle_spacing=fluid_particle_spacing, tspan=(0.0, 0.2),
              initial_fluid_size=(0.0, 0.0, 0.0),
              tank_size=(2.0, 1.0, 0.1), sound_speed=sound_speed,
              faces=(true, true, true, true, true, false),
              acceleration=(0.0, 0.0, -gravity), sphere1=sphere1, sphere2=sphere2,
              fluid_smoothing_length=fluid_smoothing_length,
              fluid_smoothing_kernel=WendlandC2Kernel{3}(),
              nu=nu, alpha=10 * nu / (fluid_smoothing_length * sound_speed),
              surface_tension_coefficient=1.0)

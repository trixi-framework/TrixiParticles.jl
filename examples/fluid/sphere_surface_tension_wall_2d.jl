# In this example we try to approach the static shape of a water droplet on a horizontal plane.
# The shape of a static droplet can be calculated from the Young-Laplace equation.
using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.0025

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 0.5)

# Boundary geometry and initial fluid particle positions
tank_size = (0.5, 0.1)

fluid_density = 1000.0
sound_speed = 120.0

sphere_radius = 0.05

sphere1_center = (0.25, sphere_radius)
sphere1 = SphereShape(fluid_particle_spacing, sphere_radius, sphere1_center,
                      fluid_density, sphere_type=VoxelSphere())

# ==========================================================================================
# ==== Fluid
fluid_smoothing_length = 1.0 * fluid_particle_spacing

# For perfect wetting
# nu = 0.0005
# For no wetting
nu = 0.001

alpha = 8 * nu / (fluid_smoothing_length * sound_speed)
# `adhesion_coefficient = 1.0` and `surface_tension_coefficient = 0.01` for perfect wetting
# `adhesion_coefficient = 0.001` and `surface_tension_coefficient = 2.0` for no wetting
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "falling_water_spheres_2d.jl"),
              sphere=nothing, sphere1=sphere1, adhesion_coefficient=0.001,
              wall_viscosity=4.0 * nu, surface_tension_coefficient=2.0, alpha=alpha,
              sound_speed=sound_speed, fluid_density=fluid_density, nu=nu,
              fluid_particle_spacing=fluid_particle_spacing, tspan=tspan,
              tank_size=tank_size)

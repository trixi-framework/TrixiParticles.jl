# ==========================================================================================
# 2D Sphere with Surface Tension Interacting with a Wall (Wetting Phenomena)
#
# This example simulates a 2D circular drop of fluid (sphere) under gravity,
# influenced by surface tension, as it interacts with a solid horizontal wall.
# The setup is designed to observe wetting phenomena (e.g., spreading or beading)
# by adjusting surface tension and adhesion parameters.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.0025

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 0.5)

# Boundary geometry and initial fluid particle positions
tank_size = (0.5, 0.1)

fluid_density = 1000.0
sound_speed = 120.0

state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1)

sphere_radius = 0.05

sphere1_center = (0.25, sphere_radius)
sphere1 = SphereShape(fluid_particle_spacing, sphere_radius, sphere1_center,
                      fluid_density, sphere_type=VoxelSphere())

# ==========================================================================================
# ==== Fluid

# Using a smoothing_length of exactly 1.0 * fluid_particle is necessary for this model to be accurate.
# This yields some numerical issues though which can be circumvented by subtracting eps().
fluid_smoothing_length = 1.0 * fluid_particle_spacing - eps()
fluid_smoothing_kernel = SchoenbergCubicSplineKernel{2}()

# For perfect wetting
# nu = 0.0005
# For no wetting
nu = 0.001

alpha = 8 * nu / (fluid_smoothing_length * sound_speed)
# `adhesion_coefficient = 1.0` and `surface_tension_coefficient = 0.01` for perfect wetting
# `adhesion_coefficient = 0.001` and `surface_tension_coefficient = 2.0` for no wetting

viscosity = ArtificialViscosityMonaghan(alpha=alpha, beta=0.0)
sphere_surface_tension = WeaklyCompressibleSPHSystem(sphere1, ContinuityDensity(),
                                                     state_equation, fluid_smoothing_kernel,
                                                     fluid_smoothing_length,
                                                     viscosity=viscosity,
                                                     acceleration=(0.0, -gravity),
                                                     surface_tension=SurfaceTensionAkinci(surface_tension_coefficient=2.0),
                                                     correction=AkinciFreeSurfaceCorrection(fluid_density),
                                                     reference_particle_spacing=fluid_particle_spacing)

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "falling_water_spheres_2d.jl"),
              sphere=nothing, sphere1=sphere1, adhesion_coefficient=0.001,
              wall_viscosity=4.0 * nu, surface_tension_coefficient=0.9, alpha=alpha,
              sound_speed=sound_speed, fluid_density=fluid_density, nu=nu,
              fluid_particle_spacing=fluid_particle_spacing, tspan=tspan,
              tank_size=tank_size, fluid_smoothing_length=fluid_smoothing_length,
              sphere_surface_tension=sphere_surface_tension)

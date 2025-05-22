# ==========================================================================================
# 3D Sphere Formation via Surface Tension
#
# This example extends the 2D sphere formation simulation (`sphere_surface_tension_2d.jl`)
# to three dimensions. It demonstrates how an initially cubical patch of fluid
# minimizes its surface area under the influence of a surface tension model,
# ideally forming a spherical shape.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ------------------------------------------------------------------------------
# Parameters Specific to 3D or Overriding 2D Defaults
# ------------------------------------------------------------------------------

fluid_density_ref_3d = 1000.0
sound_speed_3d = 20.0

# Resolution and Geometry for 3D
particle_spacing_3d = 0.15 # Coarser for 3D to manage computation time
initial_fluid_size_3d = (0.9, 0.9, 0.9)

# SPH numerical parameters for 3D
# Option 1: h = 1.0 * dp (Akinci model)
# smoothing_length_3d = 1.0 * particle_spacing_3d
# Option 2: Larger for Morris surface tension model
smoothing_length_3d = 1.5 * particle_spacing_3d
smoothing_kernel_3d = WendlandC2Kernel{3}()

# Viscosity parameter for 3D
nu_3d = 0.04

# Surface tension coefficient for the model used in the 2D file.
# This value likely needs tuning for 3D and the specific model.
# Example: if 2D uses Morris with `50 * 0.0728`
surface_tension_coeff_3d = 0.5

# Simulation time span and save interval for 3D
tspan_3d = (0.0, 100.0) # Potentially longer time for 3D to reach equilibrium
dt_save_3d = 0.25       # Saving interval for solution

# ------------------------------------------------------------------------------
# Simulation using 2D Base File
# ------------------------------------------------------------------------------
# Include the 2D sphere formation setup.
# Variables defined above will override the defaults in the 2D file.
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "sphere_surface_tension_2d.jl"),
              fluid_density_ref=fluid_density_ref_3d,
              sound_speed=sound_speed_3d,
              particle_spacing=particle_spacing_3d,
              initial_fluid_size=initial_fluid_size_3d,
              smoothing_length=smoothing_length_3d,
              smoothing_kernel=smoothing_kernel_3d,
              nu_morris=nu_3d,
              surface_tension_coefficient=surface_tension_coeff_3d,
              tspan=tspan_3d,
              dt=dt_save_3d)

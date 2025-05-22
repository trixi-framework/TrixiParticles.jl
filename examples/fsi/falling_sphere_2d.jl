# ==========================================================================================
# 2D Single Falling Sphere in Fluid (FSI)
#
# This example simulates a single elastic sphere falling through a fluid in a tank.
# ==========================================================================================

using TrixiParticles # `OrdinaryDiffEq` is loaded by the included file

# ------------------------------------------------------------------------------
# Parameters to Override from `fsi/falling_spheres_2d.jl`
# ------------------------------------------------------------------------------
# Specific parameters for a single falling sphere scenario.
# Most defaults from the base file will be used unless specified here.

# Resolution (can be different from base file default)
fluid_particle_spacing_single = 0.02

# Tank and initial fluid geometry for a potentially smaller setup
initial_fluid_size_single = (1.0, 0.9) # width, height
tank_size_single = (1.0, 1.0)          # width, height

# Simulation time span
simulation_tspan_single = (0.0, 1.0)

# Solver tolerances
abstol_single = 1e-6
reltol_single = 1e-3

# ------------------------------------------------------------------------------
# Simulation Setup using Base File
# ------------------------------------------------------------------------------
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fsi", "falling_spheres_2d.jl"), # Path to the base file
              # Disable the second solid system from the base file
              solid_system_2=nothing,
              # Override resolution and geometry
              fluid_particle_spacing=fluid_particle_spacing_single,
              solid_particle_spacing=fluid_particle_spacing_single, # Keep them same
              initial_fluid_size_default=initial_fluid_size_single, # Base file uses `_default`
              tank_size_default=tank_size_single,                 # for these geometry args
              # Override simulation time and output
              simulation_tspan=simulation_tspan_single,
              output_directory_default="out_falling_sphere_single_2d_fsi",
              # Override solver tolerances
              abstol_default=abstol_single,
              reltol_default=reltol_single
              # Note: Sphere 1 properties (radius, density, E, initial position)
              # will be taken from the defaults in `falling_spheres_2d.jl` unless
              # explicitly overridden here (e.g., `sphere1_radius=new_radius`).
              )

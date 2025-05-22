# ==========================================================================================
# 3D Hydrostatic Water Column Simulation
#
# This example sets up a 3D hydrostatic water column by including and modifying
# the 2D `hydrostatic_water_column_2d.jl` setup.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ------------------------------------------------------------------------------
# Parameters for 3D Simulation
# ------------------------------------------------------------------------------

particle_spacing_3d = 0.05

gravity_magnitude_3d = 9.81

# System acceleration vector for 3D (gravity in negative z-direction)
system_acceleration_3d = (0.0, 0.0, -gravity_magnitude_3d)

# Initial fluid and tank dimensions for 3D (width, depth, height)
initial_fluid_size_3d = (1.0, 1.0, 0.9)
tank_size_3d = (1.0, 1.0, 1.2)

# SPH smoothing kernel for 3D
smoothing_kernel_3d = SchoenbergCubicSplineKernel{3}()

# Simulation time span
tspan_3d = (0.0, 1.0)

# ------------------------------------------------------------------------------
# Simulation Setup using 2D Base File
# ------------------------------------------------------------------------------

# Include the 2D hydrostatic water column setup.
# Variables defined above will override the defaults in the 2D file.

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"),
              particle_spacing=particle_spacing_3d,
              initial_fluid_size=initial_fluid_size_3d,
              tank_size=tank_size_3d,
              system_acceleration=system_acceleration_3d,
              smoothing_kernel=smoothing_kernel_3d,
              tspan=tspan_3d)

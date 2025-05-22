# ==========================================================================================
# 3D Single Falling Sphere in Fluid (FSI)
#
# This example simulates a single elastic sphere falling through a fluid in a 3D tank.
# ==========================================================================================

using TrixiParticles # `OrdinaryDiffEq` is loaded by the included file

# ------------------------------------------------------------------------------
# Parameters to Override from `fsi/falling_spheres_2d.jl` for 3D
# ------------------------------------------------------------------------------
# Resolution (coarser for 3D for tractability)
fluid_particle_spacing_3d = 0.05

# Physical Parameters for 3D
# Gravity in y-direction as per original file's override `acceleration=(0.0, -9.81, 0.0)`
# The base file defaults to z-gravity if ndims=3 implicitly. Explicit override is better.
system_acceleration_vec_3d = (0.0, -9.81, 0.0) # Gravity in -y

# Tank and Initial Fluid Geometry for 3D (width, height, depth)
initial_fluid_size_3d = (1.0, 0.9, 1.0) # Example: height is in y-dim
tank_size_3d = (1.0, 1.0, 1.0)          # Example: height is in y-dim
# Faces for 3D tank: (x_neg, x_pos, y_neg (bottom), y_pos (top_open), z_neg, z_pos)
# Original override implies y is vertical, top is open.
tank_faces_3d = (true, true, true, false, true, true)

# Sphere 1 Initial Center for 3D (x, y, z)
sphere1_center_initial_3d = (0.5, 2.0, 0.5) # Positioned higher in y

# SPH Kernels for 3D
fluid_smoothing_kernel_3d = WendlandC2Kernel{3}()
solid_smoothing_kernel_3d = WendlandC2Kernel{3}() # Keep consistent for FSI

# Sphere Discretization Type (can be `VoxelSphere` or `RoundSphere`)
sphere_discretization_type_3d = RoundSphere() # Smoother sphere for 3D visuals

# Simulation Time Span and Output
simulation_tspan_3d = (0.0, 1.0)
output_directory_3d = "out_falling_sphere_single_3d_fsi"
output_prefix_3d = ""
# `write_meta_data=false` as per original to avoid issues with `meshio`.
write_meta_data_3d = false

# Solver Tolerances
abstol_3d = 1e-6
reltol_3d = 1e-3

# ------------------------------------------------------------------------------
# Simulation Setup using Base File
# ------------------------------------------------------------------------------
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fsi", "falling_spheres_2d.jl"), # Path to the 2D base file
              # Disable the second solid system
              solid_system_2=nothing,
              # Override resolution
              fluid_particle_spacing=fluid_particle_spacing_3d,
              solid_particle_spacing=fluid_particle_spacing_3d, # Keep them same
              # Override physical parameters for 3D
              system_acceleration_vec=system_acceleration_vec_3d,
              # Override geometry for 3D
              initial_fluid_size_default=initial_fluid_size_3d,
              tank_size_default=tank_size_3d,
              tank_faces_default=tank_faces_3d,
              # Override sphere 1 initial position for 3D
              sphere1_center_initial=sphere1_center_initial_3d,
              # Override SPH kernels for 3D
              fluid_smoothing_kernel=fluid_smoothing_kernel_3d,
              solid_smoothing_kernel=solid_smoothing_kernel_3d,
              # Override sphere discretization type
              sphere_discretization_type=sphere_discretization_type_3d,
              # Override simulation time and output
              simulation_tspan=simulation_tspan_3d,
              output_directory_default=output_directory_3d,
              output_prefix_default=output_prefix_3d,
              write_meta_data_default=write_meta_data_3d,
              # Override solver tolerances
              abstol_default=abstol_3d,
              reltol_default=reltol_3d
              # Note: Sphere 1 properties (radius, density, E) will be taken from the
              # defaults in `falling_spheres_2d.jl` unless explicitly overridden.
              )

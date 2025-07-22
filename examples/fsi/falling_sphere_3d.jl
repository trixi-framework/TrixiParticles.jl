# ==========================================================================================
# 3D Single Falling Sphere in Fluid (FSI)
#
# This example simulates a single elastic sphere falling into a fluid in a 3D tank.
# ==========================================================================================

using TrixiParticles

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fsi", "falling_spheres_2d.jl"),
              solid_system_2=nothing, fluid_particle_spacing=0.05,
              initial_fluid_size=(1.0, 0.9, 1.0), tank_size=(1.0, 1.0, 1.0),
              faces=(true, true, true, false, true, true),
              acceleration=(0.0, -9.81, 0.0), sphere1_center=(0.5, 2.0, 0.5),
              fluid_smoothing_kernel=WendlandC2Kernel{3}(),
              solid_smoothing_kernel=WendlandC2Kernel{3}(),
              sphere_type=RoundSphere(),
              output_directory="out", prefix="",
              write_meta_data=false, # Files with meta data can't be read by meshio
              tspan=(0.0, 1.0), abstol=1e-6, reltol=1e-3)

# ==========================================================================================
# 3D Falling Water Spheres Simulation (With and Without Surface Tension)
#
# This example extends `falling_water_spheres_2d.jl` to three dimensions.
# It simulates two spherical volumes of water falling under gravity.
# One sphere includes a surface tension model, while the other does not,
# demonstrating the effect of surface tension in 3D.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ------------------------------------------------------------------------------
# Parameters Specific to 3D or Overriding 2D Defaults
# ------------------------------------------------------------------------------
# Particle spacing (resolution)
fluid_particle_spacing = 0.005

# Physical properties
gravity = -9.81
gravity_vec_3d = (0.0, 0.0, gravity)
physical_nu_3d = 0.001
fluid_density_3d = 1000.0
sound_speed_3d = 50.0

# Sphere properties
sphere_radius_3d = 0.05
initial_velocity_spheres_3d = (0.0, 0.0, -1.0) # Falling in -z direction

# Define initial positions for the two spheres in 3D
sphere1_center_3d = (0.5, 0.5, 0.075)
sphere2_center_3d = (1.5, 0.5, 0.075)

# Create 3D sphere particle sets
sphere1_particles_3d = SphereShape(fluid_particle_spacing, sphere_radius_3d,
                                   sphere1_center_3d, fluid_density_3d,
                                   sphere_type=VoxelSphere(),
                                   velocity=initial_velocity_spheres_3d)

sphere2_particles_3d = SphereShape(fluid_particle_spacing, sphere_radius_3d,
                                   sphere2_center_3d, fluid_density_3d,
                                   sphere_type=VoxelSphere(),
                                   velocity=initial_velocity_spheres_3d)

fluid_smoothing_length_3d = 1.0 * fluid_particle_spacing
fluid_smoothing_kernel_3d = SchoenbergCubicSplineKernel{3}()

# Monaghan artificial viscosity alpha parameter for 3D
alpha_viscosity_3d = 10 * physical_nu_3d / (fluid_smoothing_length_3d * sound_speed_3d)

# Surface tension model for the first sphere in 3D
surface_tension_model_akinci_3d = SurfaceTensionAkinci(surface_tension_coefficient=0.05)

# Simulation time span for 3D
tspan_3d = (0.0, 0.1)

# Tank dimensions for 3D (acts as a floor and side walls, open top)
# (width, depth, height)
tank_dims_3d = (2.0, 1.0, 0.1)
# `faces` for `RectangularTank` in 3D: (x_neg, x_pos, y_neg, y_pos, z_neg, z_pos)
# Here: floor (z_neg=true) and four side walls, open top (z_pos=false).
tank_faces_3d = (true, true, true, true, true, false)

# ------------------------------------------------------------------------------
# Load and Adapt 2D Falling Spheres Example for 3D
# ------------------------------------------------------------------------------
# We include the 2D example file. Most objects like `state_equation`,
# `fluid_density_calculator`, `boundary_density_calculator`, etc., will be
# created by that file, using the parameters overridden here.
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "falling_water_spheres_2d.jl"),
              # Override basic parameters for 3D
              fluid_particle_spacing=fluid_particle_spacing,
              tspan=tspan_3d,
              gravity_vec=gravity_vec_3d,
              tank_initial_fluid_size=(0.0, 0.0, 0.0),
              tank_dims=tank_dims_3d,
              sound_speed=sound_speed_3d,
              faces=tank_faces_3d,
              sphere1_particles=sphere1_particles_3d,
              sphere2_particles=sphere2_particles_3d,
              fluid_smoothing_length=fluid_smoothing_length_3d,
              fluid_smoothing_kernel=fluid_smoothing_kernel_3d,
              physical_nu=physical_nu_3d,
              alpha_viscosity=alpha_viscosity_3d,
              surface_tension_model_akinci=surface_tension_model_akinci_3d
              )

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

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------
# Resolution
fluid_particle_spacing = 0.0025 # High resolution for better surface effects

# Physical parameters
gravity_magnitude = 9.81
gravity_vec = (0.0, -gravity_magnitude)
simulation_tspan = (0.0, 0.5)

# Fluid properties
fluid_density_ref = 1000.0
sound_speed_fluid = 120.0 # High sound speed for stability with strong surface forces

fluid_state_equation = StateEquationCole(sound_speed=sound_speed_fluid,
                                         reference_density=fluid_density_ref,
                                         exponent=1)

# Initial sphere (drop) geometry
sphere_radius_initial = 0.05
# Position the sphere initially touching or slightly above the tank floor.
# Tank floor is at y=0.
sphere_center_initial = (0.25, sphere_radius_initial) # x, y

# Tank dimensions (acts as a floor for the sphere)
tank_domain_size = (0.5, 0.1) # width, height

# ------------------------------------------------------------------------------
# Wetting Behavior Parameters
# ------------------------------------------------------------------------------
# These parameters determine if the fluid wets the surface (spreads out) or
# beads up (no wetting). Values are empirical and model-dependent.

# Option 1: "Perfect Wetting" (fluid spreads
# nu_wetting = 0.0005
# adhesion_coeff_wetting = 1.0
# surface_tension_coeff_wetting = 0.01 # Lower surface tension relative to adhesion

# Option 2: "No Wetting" (fluid beads up)
nu_no_wetting = 0.001 # Kinematic viscosity for ArtificialViscosityMonaghan
adhesion_coeff_no_wetting = 0.001 # Low adhesion to the wall
surface_tension_coeff_no_wetting = 2.0  # Higher surface tension

# ------------------------------------------------------------------------------
# Fluid System Setup
# ------------------------------------------------------------------------------
# Create the initial fluid sphere
fluid_sphere_particles = SphereShape(fluid_particle_spacing, sphere_radius_initial,
                                     sphere_center_initial, fluid_density_ref,
                                     sphere_type=VoxelSphere())

# Subtracting eps() can help with numerical precision at kernel boundaries.
fluid_smoothing_length_sphere = 1.0 * fluid_particle_spacing - eps()
fluid_smoothing_kernel_sphere = SchoenbergCubicSplineKernel{2}()

# Viscosity model for the fluid sphere
alpha_monaghan_sphere = 8 * nu_no_wetting / (fluid_smoothing_length_sphere * sound_speed_fluid)
viscosity_sphere_model = ArtificialViscosityMonaghan(alpha=alpha_monaghan_sphere, beta=0.0)

# Surface tension model (Akinci et al.) and free-surface correction
surface_tension_model_sphere = SurfaceTensionAkinci(surface_tension_coefficient=surface_tension_coeff_no_wetting)
free_surface_correction_sphere = AkinciFreeSurfaceCorrection(fluid_density_ref)

# Define the fluid system for the sphere
sphere_fluid_system = WeaklyCompressibleSPHSystem(fluid_sphere_particles,
                                                  ContinuityDensity(),
                                                  fluid_state_equation,
                                                  fluid_smoothing_kernel_sphere,
                                                  fluid_smoothing_length_sphere,
                                                  viscosity=viscosity_sphere_model,
                                                  acceleration=gravity_vec,
                                                  surface_tension=surface_tension_model_sphere,
                                                  correction=free_surface_correction_sphere,
                                                  reference_particle_spacing=fluid_particle_spacing)

# ------------------------------------------------------------------------------
# Boundary System Setup (Tank Floor) - Leverages `falling_water_spheres_2d.jl`
# ------------------------------------------------------------------------------
# The `falling_water_spheres_2d.jl` example sets up a `RectangularTank` and a
# `BoundaryModelDummyParticles`. We will include it and override parameters.

# Wall viscosity for interaction with the sphere.
# This can influence how the fluid behaves near the wall (e.g., shear).
wall_kinematic_viscosity = 4.0 * nu_no_wetting

# `sphere1_system_surftens=sphere_fluid_system`: We replace one of the sphere systems
#                                               from the included file with our defined system.
# `sphere2_system_nosurftens=nothing`: We disable the second sphere from the included file.
# Other parameters are overridden to match this specific "wetting" scenario.
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "falling_water_spheres_2d.jl"),
              sphere1_system_surftens=sphere_fluid_system,
              sphere2_system_nosurftens=nothing,
              tank_dims=tank_domain_size,
              fluid_particle_spacing=fluid_particle_spacing,
              tspan=simulation_tspan,
              gravity_vec=gravity_vec,
              state_equation=fluid_state_equation,
              sound_speed=sound_speed_fluid,
              fluid_density=fluid_density_ref,
              fluid_smoothing_length=fluid_smoothing_length_sphere,
              fluid_smoothing_kernel=fluid_smoothing_kernel_sphere,
              adhesion_coefficient=adhesion_coeff_no_wetting,
              wall_kinematic_viscosity=wall_kinematic_viscosity
              )

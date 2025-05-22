# ==========================================================================================
# 2D Falling Spheres in Fluid (FSI) - Base Setup
#
# This file provides a base setup for simulating one or two elastic spheres
# falling through a fluid in a tank.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ------------------------------------------------------------------------------
# Parameters (Defaults, can be overridden by `trixi_include`)
# ------------------------------------------------------------------------------
# Resolution Parameters
fluid_particle_spacing = 0.02 # meters
solid_particle_spacing = fluid_particle_spacing # Keep solid and fluid spacing same by default

# Tank boundary particle layers and spacing ratio
tank_boundary_layers = 3
tank_spacing_ratio = 1

# Physical Parameters
gravity_magnitude = 9.81
# `system_acceleration_vec` can be overridden for 3D or different gravity direction.
system_acceleration_vec = (0.0, -gravity_magnitude)
simulation_tspan = (0.0, 1.0) # seconds

# Fluid Properties
fluid_density_ref = 1000.0 # kg/m^3
initial_fluid_height_ref = 0.9 # Reference height for sound speed calculation (if `initial_fluid_size` is used)
sound_speed_fluid = 10 * sqrt(gravity_magnitude * initial_fluid_height_ref)
# Exponent 1 for state equation is common in FSI.
fluid_state_equation = StateEquationCole(sound_speed=sound_speed_fluid,
                                         reference_density=fluid_density_ref,
                                         exponent=1)

# Solid Sphere Properties (Defaults for Sphere 1 and Sphere 2)
# Sphere 1
sphere1_radius = 0.3 # meters
sphere1_density = 500.0 # kg/m^3 (lighter than fluid)
sphere1_youngs_modulus_E = 7.0e4 # Pa
sphere1_center_initial = (0.5, 1.6) # x, y meters

# Sphere 2 (can be disabled by passing `solid_system_2=nothing` via `trixi_include`)
sphere2_radius = 0.2 # meters
sphere2_density = 1100.0 # kg/m^3 (heavier than fluid)
sphere2_youngs_modulus_E = 1.0e5 # Pa
sphere2_center_initial = (1.5, 1.6) # x, y meters

# Common Poisson's ratio for spheres
poissons_ratio_nu_spheres = 0.0

# Sphere discretization type (`VoxelSphere` or `RoundSphere`)
# `VoxelSphere` creates a more "blocky" sphere from cubic cells.
# `RoundSphere` creates a smoother particle distribution for the sphere.
sphere_discretization_type = VoxelSphere()

# Tank and Initial Fluid Geometry
# `initial_fluid_size` defines the extent of the quiescent fluid.
# `tank_size` defines the overall domain for tank walls.
initial_fluid_size_default = (2.0, 0.9) # width, height
tank_size_default = (2.0, 1.0)        # width, height
# `tank_faces` defines which walls are present (e.g., open top).
# Default for 2D: (left, right, bottom, top_open)
tank_faces_default = (true, true, true, false)

# Callbacks and Output
output_directory_default = "out_falling_spheres_fsi_2d"
output_prefix_default = ""
write_meta_data_default = true

# ------------------------------------------------------------------------------
# Experiment Setup: Tank, Fluid, and Solid Spheres
# ------------------------------------------------------------------------------
# Tank and Initial Fluid
tank_setup = RectangularTank(fluid_particle_spacing, initial_fluid_size_default,
                             tank_size_default, fluid_density_ref,
                             n_layers=tank_boundary_layers,
                             spacing_ratio=tank_spacing_ratio,
                             faces=tank_faces_default,
                             acceleration=system_acceleration_vec, # For hydrostatic init
                             state_equation=fluid_state_equation)

# Solid Sphere 1 Particles
sphere1_particles = SphereShape(solid_particle_spacing, sphere1_radius,
                                sphere1_center_initial, sphere1_density,
                                sphere_type=sphere_discretization_type)

# Solid Sphere 2 Particles (created only if `solid_system_2` is not `nothing` later)
sphere2_particles = SphereShape(solid_particle_spacing, sphere2_radius,
                                sphere2_center_initial, sphere2_density,
                                sphere_type=sphere_discretization_type)

# ------------------------------------------------------------------------------
# Fluid System Setup (Weakly Compressible SPH)
# ------------------------------------------------------------------------------
fluid_smoothing_length = 1.5 * fluid_particle_spacing
# `fluid_smoothing_kernel` can be overridden for 3D.
fluid_smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()
fluid_viscosity_model = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)
# Density diffusion can help stabilize the fluid density field.
fluid_density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)

fluid_system = WeaklyCompressibleSPHSystem(tank_setup.fluid, fluid_density_calculator,
                                           fluid_state_equation, fluid_smoothing_kernel,
                                           fluid_smoothing_length,
                                           viscosity=fluid_viscosity_model,
                                           density_diffusion=fluid_density_diffusion,
                                           acceleration=system_acceleration_vec,
                                           reference_particle_spacing=fluid_particle_spacing)

# ------------------------------------------------------------------------------
# Boundary System Setup (Tank Walls)
# ------------------------------------------------------------------------------
# Using BernoulliPressureExtrapolation as per original file for this specific setup.
# AdamiPressureExtrapolation is also a common choice.
tank_boundary_density_calculator = BernoulliPressureExtrapolation()
tank_boundary_model = BoundaryModelDummyParticles(tank_setup.boundary.density,
                                                  tank_setup.boundary.mass,
                                                  fluid_state_equation,
                                                  tank_boundary_density_calculator,
                                                  fluid_smoothing_kernel, fluid_smoothing_length,
                                                  reference_particle_spacing=fluid_particle_spacing)
tank_boundary_system = BoundarySPHSystem(tank_setup.boundary, tank_boundary_model)

# ------------------------------------------------------------------------------
# Solid System Setup (Elastic Spheres - Total Lagrangian SPH)
# ------------------------------------------------------------------------------
# Smoothing length for solids (can differ from fluid).
solid_smoothing_length_spheres = sqrt(2) * solid_particle_spacing
# `solid_smoothing_kernel` can be overridden for 3D.
solid_smoothing_kernel = WendlandC2Kernel{2}()

# Penalty force for inter-solid contact (e.g., sphere-sphere if applicable, or self-contact)
# and solid-boundary contact if not handled by FSI pressure.
solid_penalty_force = PenaltyForceGanzenmueller(alpha=0.3) # Alpha controls stiffness

# --- Solid System 1 (Sphere 1) ---
# Hydrodynamic properties for FSI boundary model on sphere 1
hydrodynamic_densities_sphere1 = fluid_density_ref .* ones(size(sphere1_particles.density))
hydrodynamic_masses_sphere1 = hydrodynamic_densities_sphere1 .* solid_particle_spacing^ndims(fluid_system) # Use ndims for 2D/3D

solid_fsi_boundary_model_sphere1 = BoundaryModelDummyParticles(hydrodynamic_densities_sphere1,
                                                               hydrodynamic_masses_sphere1,
                                                               fluid_state_equation,
                                                               tank_boundary_density_calculator, # Use same type as tank
                                                               fluid_smoothing_kernel, fluid_smoothing_length,
                                                               reference_particle_spacing=fluid_particle_spacing)

solid_system_1 = TotalLagrangianSPHSystem(sphere1_particles,
                                          solid_smoothing_kernel, solid_smoothing_length_spheres,
                                          sphere1_youngs_modulus_E, poissons_ratio_nu_spheres,
                                          boundary_model=solid_fsi_boundary_model_sphere1,
                                          penalty_force=solid_penalty_force,
                                          acceleration=system_acceleration_vec,
                                          reference_particle_spacing=solid_particle_spacing)

# --- Solid System 2 (Sphere 2) ---
# This system is optional and can be disabled by passing `solid_system_2=nothing`
# via `trixi_include`.
solid_system_2 =وشنsolid_system_2 # Placeholder, will be defined if not `nothing`
if !isnothing(solid_system_2) # Check if it should be created (default is to create)
    hydrodynamic_densities_sphere2 = fluid_density_ref .* ones(size(sphere2_particles.density))
    hydrodynamic_masses_sphere2 = hydrodynamic_densities_sphere2 .* solid_particle_spacing^ndims(fluid_system)

    solid_fsi_boundary_model_sphere2 = BoundaryModelDummyParticles(hydrodynamic_densities_sphere2,
                                                                   hydrodynamic_masses_sphere2,
                                                                   fluid_state_equation,
                                                                   tank_boundary_density_calculator,
                                                                   fluid_smoothing_kernel, fluid_smoothing_length,
                                                                   reference_particle_spacing=fluid_particle_spacing)

    solid_system_2 = TotalLagrangianSPHSystem(sphere2_particles,
                                              solid_smoothing_kernel, solid_smoothing_length_spheres,
                                              sphere2_youngs_modulus_E, poissons_ratio_nu_spheres,
                                              boundary_model=solid_fsi_boundary_model_sphere2,
                                              penalty_force=solid_penalty_force,
                                              acceleration=system_acceleration_vec,
                                              reference_particle_spacing=solid_particle_spacing)
else
    # Ensure `solid_system_2` is explicitly `nothing` if the keyword was passed.
    # This handles the case where the `solid_system_2` keyword is passed as `nothing`.
    solid_system_2 = nothing
end


# ------------------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------------------
# Collect all active systems.
active_systems = (fluid_system, tank_boundary_system, solid_system_1)
if !isnothing(solid_system_2)
    active_systems = (active_systems..., solid_system_2)
end

semi = Semidiscretization(active_systems..., # Splat the tuple of systems
                          parallelization_backend=PolyesterBackend())
ode = semidiscretize(semi, simulation_tspan)

# Callbacks
info_callback = InfoCallback(interval=10) # Frequent info for FSI debugging
saving_callback = SolutionSavingCallback(dt=0.02,
                                         output_directory=output_directory_default,
                                         prefix=output_prefix_default,
                                         write_meta_data=write_meta_data_default)
callbacks = CallbackSet(info_callback, saving_callback)

# Default solver tolerances (can be overridden by `trixi_include`)
abstol_default = 1e-6
reltol_default = 1e-3

# Solve the ODE system.
sol = solve(ode, RDPK3SpFSAL49(), # Solver from original file
            abstol=abstol_default,
            reltol=reltol_default,
            save_everystep=false,
            callback=callbacks)

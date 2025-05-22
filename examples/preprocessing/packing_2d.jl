# ==========================================================================================
# 2D Particle Packing within a Complex Geometry
#
# This example demonstrates how to:
# 1. Load a 2D geometry (e.g., a circle defined by a boundary curve).
# 2. Generate an initial, potentially overlapping, distribution of "fluid" particles
#    inside the geometry and "boundary" particles forming a layer around it.
# 3. Use the `ParticlePackingSystem` to run a pseudo-SPH simulation that relaxes
#    the particle positions, achieving a more uniform and non-overlapping distribution.
# 4. Visualize the initial and packed particle configurations.
#
# This is a common preprocessing step to create stable initial conditions for SPH simulations.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq
using Plots # For final visualization

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------
# Geometry file
geometry_filename_stem_pack = "circle"
geometry_file_path_pack = joinpath(pkgdir(TrixiParticles), "examples", "preprocessing", "data",
                                   geometry_filename_stem_pack * ".asc")

# Resolution and Boundary Thickness
particle_spacing_pack = 0.1 # meters
# `boundary_packing_thickness` defines the thickness of the layer where boundary
# particles will be sampled and packed.
boundary_packing_thickness = 5 * particle_spacing_pack

# Particle Packing Parameters
# `tlsph=false` is a flag often related to how boundary particles are handled or initialized.
# For packing, its specific effect might depend on `ParticlePackingSystem` internals.
use_tlsph_packing_flag = false # As per original example

# Background pressure for the packing algorithm.
# Scales the pseudo-forces driving particles apart. A higher value can speed up
# packing convergence but might require smaller time steps.
background_pressure_packing = 1.0 # Unitless scaling factor

# Smoothing length for the packing algorithm, typically slightly less than particle spacing.
smoothing_length_packing = 0.8 * particle_spacing_pack

# Boundary compression factor for `ParticlePackingSystem` for boundary particles.
# Controls how strongly boundary particles are pushed towards their ideal positions.
boundary_compression_factor = 0.8

# Simulation control for packing
max_iterations_packing = 1000
steady_state_abstol = 1.0e-7 # Absolute tolerance for kinetic energy to detect steady state
steady_state_reltol = 1.0e-6 # Relative tolerance

# ------------------------------------------------------------------------------
# Load Geometry and Initial Particle Sampling
# ------------------------------------------------------------------------------
particle_density_pack = 1.0 # Dummy density for packing visualization

println("Loading geometry for 2D packing: $geometry_file_path_pack")
geometry_pack = load_geometry(geometry_file_path_pack)

# Create a Signed Distance Field (SDF) from the geometry.
# The SDF is used by the packing algorithm to keep particles within the desired regions
# and to sample boundary particles accurately.
# `use_for_boundary_packing=true` and `max_signed_distance` are important for boundary sampling.
sdf_pack = SignedDistanceField(geometry_pack, particle_spacing_pack;
                               use_for_boundary_packing=true,
                               max_signed_distance=boundary_packing_thickness)

# Algorithm to determine if initial "fluid" sample points are inside the geometry.
point_in_geometry_test_pack = WindingNumberJacobson(geometry=geometry_pack,
                                                    winding_number_factor=0.4, # Default
                                                    hierarchical_winding=true)

# Initial sampling of "fluid" particles inside the geometry.
# These particles will be relaxed by the packing algorithm.
println("Initial sampling of fluid particles for packing...")
fluid_particles_initial_sample = ComplexShape(geometry_pack;
                                              particle_spacing=particle_spacing_pack,
                                              density=particle_density_pack,
                                              point_in_geometry_algorithm=point_in_geometry_test_pack)
# Result is `fluid_particles_initial_sample.initial_condition`

# Initial sampling of boundary particles using the SDF.
# These particles form a fixed layer representing the geometry's boundary.
println("Initial sampling of boundary particles for packing...")
boundary_particles_initial_sample = sample_boundary(sdf_pack;
                                                    boundary_density=particle_density_pack,
                                                    boundary_thickness=boundary_packing_thickness,
                                                    tlsph=use_tlsph_packing_flag)

# Export initial (unpacked) particle distributions.
trixi2vtk(fluid_particles_initial_sample.initial_condition,
          filename="out/$(geometry_filename_stem_pack)_fluid_unpacked.vtp")
trixi2vtk(boundary_particles_initial_sample,
          filename="out/$(geometry_filename_stem_pack)_boundary_unpacked.vtp")
println("Exported unpacked fluid and boundary particles to 'out/' directory.")

# ------------------------------------------------------------------------------
# Setup Particle Packing Systems
# ------------------------------------------------------------------------------
# System for the "fluid" particles that will be packed.
fluid_packing_system = ParticlePackingSystem(fluid_particles_initial_sample.initial_condition;
                                             smoothing_length=smoothing_length_packing,
                                             signed_distance_field=sdf_pack,
                                             tlsph=use_tlsph_packing_flag,
                                             background_pressure=background_pressure_packing)

# System for the boundary particles. These are also "packed" to ensure they
# correctly represent the boundary, but their movement might be more constrained.
# `is_boundary=true` and `boundary_compress_factor` are specific to this.
boundary_packing_system = ParticlePackingSystem(boundary_particles_initial_sample;
                                                smoothing_length=smoothing_length_packing,
                                                is_boundary=true,
                                                signed_distance_field=sdf_pack,
                                                tlsph=use_tlsph_packing_flag,
                                                boundary_compress_factor=boundary_compression_factor,
                                                background_pressure=background_pressure_packing)

# ------------------------------------------------------------------------------
# Run Particle Packing Simulation
# ------------------------------------------------------------------------------
semi_packing = Semidiscretization(fluid_packing_system, boundary_packing_system)

# Use a large tspan; termination is controlled by `maxiters` or `SteadyStateReachedCallback`.
packing_tspan = (0.0, 10.0)
ode_packing = semidiscretize(semi_packing, packing_tspan)

# Callbacks for the packing simulation
# `SteadyStateReachedCallback` stops the simulation when particle movement (kinetic energy) is minimal.
steady_state_cb_packing = SteadyStateReachedCallback(; interval=10, interval_size=200,
                                                     abstol=steady_state_abstol,
                                                     reltol=steady_state_reltol)
info_cb_packing = InfoCallback(interval=50) # Print progress

# Optional: Save intermediate packing states.
enable_interval_saving_packing = false # Set to true to save intermediate steps
saving_cb_packing = enable_interval_saving_packing ?
                    SolutionSavingCallback(interval=10, prefix="out/packing_step",
                                           kinetic_energy=kinetic_energy) : # Track kinetic energy
                    nothing

# Track kinetic energy to monitor convergence (alternative to SteadyStateReachedCallback output).
kinetic_energy_cb_packing = PostprocessCallback(; kinetic_energy=kinetic_energy, interval=1,
                                                filename="out/packing_kinetic_energy",
                                                write_file_interval=50)

# `UpdateCallback` might be needed if neighborhood search requires updates.
callbacks_packing = CallbackSet(UpdateCallback(), saving_cb_packing,
                                info_cb_packing, steady_state_cb_packing,
                                kinetic_energy_cb_packing)

println("Starting particle packing simulation...")
sol_packing = solve(ode_packing, RDPK3SpFSAL35(); # Suitable solver for SPH-like systems
                    save_everystep=false,
                    maxiters=max_iterations_packing,
                    callback=callbacks_packing,
                    dtmax=1e-2) # Limit max timestep for stability

# ------------------------------------------------------------------------------
# Extract and Export Packed Particle Configurations
# ------------------------------------------------------------------------------
# Create `InitialCondition` objects from the final state of the packing simulation.
packed_fluid_ic = InitialCondition(sol_packing, fluid_packing_system, semi_packing)
packed_boundary_ic = InitialCondition(sol_packing, boundary_packing_system, semi_packing)

# Export packed particle distributions.
trixi2vtk(packed_fluid_ic,
          filename="out/$(geometry_filename_stem_pack)_fluid_packed.vtp")
trixi2vtk(packed_boundary_ic,
          filename="out/$(geometry_filename_stem_pack)_boundary_packed.vtp")
println("Exported packed fluid and boundary particles to 'out/' directory.")

# ------------------------------------------------------------------------------
# Visualize Initial vs. Packed Configurations
# ------------------------------------------------------------------------------
# Create a Plots.Shape object from the loaded geometry for overlay.
geometry_plot_shape = Plots.Shape(stack(geometry_pack.vertices)[1, :],
                                  stack(geometry_pack.vertices)[2, :])

plot_comparison = plot(layout=(1, 2), size=(900, 450), aspect_ratio=:equal)

# Subplot 1: Initial (unpacked) fluid particles
plot!(plot_comparison[1], fluid_particles_initial_sample.initial_condition,
      markerstrokewidth=0.5, markersize=3, label="Unpacked Fluid",
      title="Initial (Unpacked) State")
plot!(plot_comparison[1], geometry_plot_shape, linecolor=:black, fillalpha=0,
      linewidth=1.5, label="Geometry Boundary")
plot!(plot_comparison[1], boundary_particles_initial_sample, color=:gray,
      markersize=2, markerstrokewidth=0, label="Unpacked Boundary")


# Subplot 2: Packed fluid particles
plot!(plot_comparison[2], packed_fluid_ic,
      markerstrokewidth=0.5, markersize=3, label="Packed Fluid",
      title="Final (Packed) State")
plot!(plot_comparison[2], geometry_plot_shape, linecolor=:black, fillalpha=0,
      linewidth=1.5, label="Geometry Boundary")
plot!(plot_comparison[2], packed_boundary_ic, color=:gray,
      markersize=2, markerstrokewidth=0, label="Packed Boundary")


display(plot_comparison)
println("Particle packing 2D example finished. Check 'out/' directory for VTK files and plot.")

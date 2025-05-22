# ==========================================================================================
# 3D Particle Packing within a Complex Geometry (e.g., Sphere STL)
#
# This example demonstrates 3D particle packing by leveraging the 2D packing script
# (`packing_2d.jl`). It loads a 3D geometry (STL file), sets 3D-specific parameters,
# and then includes the 2D script, which will perform the packing logic adapted
# for 3D.
# ==========================================================================================

using TrixiParticles
# `OrdinaryDiffEq` and `Plots` will be loaded by the included `packing_2d.jl` if needed by it.

# ------------------------------------------------------------------------------
# Parameters for 3D Packing (Overrides for 2D Defaults)
# ------------------------------------------------------------------------------
# Geometry file for 3D (e.g., an STL of a sphere)
geometry_filename_stem_3d_pack = "sphere"
geometry_file_path_3d_pack = joinpath(pkgdir(TrixiParticles), "examples", "preprocessing", "data",
                                      geometry_filename_stem_3d_pack * ".stl")

# Resolution and Boundary Thickness for 3D
particle_spacing_3d_pack = 0.1 # meters
# Thickness of the boundary layer for particle sampling and packing.
# For 3D, a thicker layer might be needed for good coverage.
boundary_packing_thickness_3d = 8 * particle_spacing_3d_pack # Increased thickness for 3D

# Particle density (dummy value for packing visualization)
particle_density_3d_pack = 1000.0 # Consistent with typical fluid densities

# `tlsph` flag (see `packing_2d.jl` for its role)
use_tlsph_packing_flag_3d = true # Original example set this to true for 3D

# Option for saving intermediate packing intervals (can be slow for 3D)
enable_interval_saving_3d_packing = false

# ------------------------------------------------------------------------------
# Include 2D Packing Script with 3D Overrides
# ------------------------------------------------------------------------------
# The `packing_2d.jl` script contains the core logic for geometry loading,
# initial sampling, setting up `ParticlePackingSystem`, and running the simulation.
# TrixiParticles.jl handles dimensionality (2D vs. 3D) based on the input geometry
# and SPH kernel choices (which are usually dimension-generic or selected based on `ndims`).
#
# We override key parameters from `packing_2d.jl` to suit this 3D case.
println("Starting 3D particle packing using base 2D script...")
trixi_include(@__MODULE__, # Ensure correct module context for included file
              joinpath(examples_dir(), "preprocessing", "packing_2d.jl"),
              # Override geometry and resolution
              geometry_file_path_pack=geometry_file_path_3d_pack, # Path to the 3D STL file
              particle_spacing_pack=particle_spacing_3d_pack,
              boundary_packing_thickness=boundary_packing_thickness_3d,
              # Override particle properties
              particle_density_pack=particle_density_3d_pack,
              # Override packing flags and control
              use_tlsph_packing_flag=use_tlsph_packing_flag_3d,
              enable_interval_saving_packing=enable_interval_saving_3d_packing,
              # Override output naming stems for clarity
              geometry_filename_stem_pack=geometry_filename_stem_3d_pack * "_3d",
              # Note: `background_pressure_packing`, `smoothing_length_packing` (relative to dp),
              # `boundary_compression_factor`, `max_iterations_packing`, and steady state
              # tolerances will be taken from `packing_2d.jl` defaults unless overridden here.
              # The plotting part of `packing_2d.jl` is 2D-specific and will likely error or
              # produce an unsuitable plot for 3D. For 3D visualization, one would typically
              # rely on the exported VTK/VTP files and tools like ParaView.
              # To prevent the 2D plot from running, you might need to add a conditional
              # or a `run_plotting=false` override if the 2D script supports it.
              # For now, we let it run and expect a potential plotting issue for the 3D case from 2D script.
              plot_comparison=nothing # Attempt to disable plotting if `packing_2d.jl` checks this
              )

println("\n3D particle packing example finished.")
println("Check 'out/' directory for VTK/VTP files (e.g., $(geometry_filename_stem_3d_pack)_3d_fluid_packed.vtp).")
println("Note: 2D plotting from the included script might not be meaningful for 3D output.")

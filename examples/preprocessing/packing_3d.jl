# ==========================================================================================
# 3D Particle Packing within a Complex Geometry (e.g., Sphere STL)
#
# This example demonstrates 3D particle packing by leveraging the 2D packing script
# (`packing_2d.jl`). It loads a 3D geometry (STL file), sets 3D-specific parameters,
# and then includes the 2D script, which will perform the packing logic adapted
# for 3D.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

filename = "sphere"
file = pkgdir(TrixiParticles, "examples", "preprocessing", "data", filename * ".stl")

# ==========================================================================================
# ==== Resolution
particle_spacing = 0.1

# The following depends on the sampling of the particles. In this case `boundary_thickness`
# means literally the thickness of the boundary packed with boundary particles and *not*
# how many rows of boundary particles will be sampled.
boundary_thickness = 8 * particle_spacing

trixi_include(joinpath(examples_dir(), "preprocessing", "packing_2d.jl"),
              density=1000.0, particle_spacing=particle_spacing, file=file,
              boundary_thickness=boundary_thickness, tlsph=true,
              save_intervals=false)

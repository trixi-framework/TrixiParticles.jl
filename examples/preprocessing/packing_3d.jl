using TrixiParticles
using OrdinaryDiffEq

filename = "sphere"
file = joinpath("examples", "preprocessing", "data", filename * ".stl")

# ==========================================================================================
# ==== Packing parameters
tlsph = true
maxiters = 100
save_intervals = false

# ==========================================================================================
# ==== Resolution
particle_spacing = 0.1

# The following depends on the sampling of the particles. In this case `boundary_thickness`
# means literally the thickness of the boundary packed with boundary particles and *not*
# how many rows of boundary particles will be sampled.
boundary_thickness = 8particle_spacing

trixi_include(joinpath(examples_dir(), "preprocessing", "packing_2d.jl"),
              density=1000.0, particle_spacing=particle_spacing, file=file,
              boundary_thickness=boundary_thickness, tlsph=tlsph,
              save_intervals=save_intervals)

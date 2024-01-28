include("dummy_particles/dummy_particles.jl")
include("open_boundary/system.jl")
include("system.jl")
# Monaghan-Kajtar repulsive boundary particles require the `BoundarySPHSystem`
# and the `TotalLagrangianSPHSystem` and are therefore included later.

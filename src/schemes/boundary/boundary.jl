include("open_boundary/boundary_zones.jl")
include("open_boundary/mirroring.jl")
include("open_boundary/method_of_characteristics.jl")
include("open_boundary/system.jl")
include("open_boundary/dynamical_pressure.jl")
include("dummy_particles/dummy_particles.jl")
include("system.jl")
# Monaghan-Kajtar repulsive boundary particles require the `BoundarySPHSystem`
# and the `TotalLagrangianSPHSystem` and are therefore included later.

@inline Base.ndims(boundary_model::BoundaryModelDummyParticles) = ndims(boundary_model.smoothing_kernel)

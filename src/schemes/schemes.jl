# Include all schemes without rhs first. The rhs depends on the systems to define
# interactions between the different system types.
include("fluid/fluid.jl")
include("boundary/boundary.jl")
# Density diffusion requires `OpenBoundarySystem`
include("fluid/weakly_compressible_sph/density_diffusion.jl")
include("structure/total_lagrangian_sph/total_lagrangian_sph.jl")
include("structure/discrete_element_method/discrete_element_method.jl")
# Monaghan-Kajtar repulsive boundary particles require the `WallBoundarySystem`
# and the `TotalLagrangianSPHSystem`.
include("boundary/wall_boundary/monaghan_kajtar.jl")
# Implicit incompressible SPH requires the `BoundarySPHSystem`
include("fluid/implicit_incompressible_sph/implicit_incompressible_sph.jl")

# Include rhs for all schemes
include("fluid/weakly_compressible_sph/rhs.jl")
include("fluid/entropically_damped_sph/rhs.jl")
include("fluid/implicit_incompressible_sph/rhs.jl")
include("boundary/wall_boundary/rhs.jl")
include("structure/total_lagrangian_sph/rhs.jl")
include("structure/discrete_element_method/rhs.jl")

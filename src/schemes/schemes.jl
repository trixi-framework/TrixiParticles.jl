# Include open boundary system first,
# since this system should not depend on specific systems.
include("boundary/open_boundary/system.jl")

# Include all schemes without rhs first. The rhs depends on the systems to define
# interactions between the different system types.
include("fluid/fluid.jl")
include("boundary/boundary.jl")
include("solid/total_lagrangian_sph/total_lagrangian_sph.jl")
# Monaghan-Kajtar repulsive boundary particles require the `BoundarySPHSystem`
# and the `TotalLagrangianSPHSystem`.
include("boundary/monaghan_kajtar/monaghan_kajtar.jl")

# Include rhs for all schemes
include("fluid/weakly_compressible_sph/rhs.jl")
include("fluid/entropically_damped_sph/rhs.jl")
include("boundary/rhs.jl")
include("solid/total_lagrangian_sph/rhs.jl")

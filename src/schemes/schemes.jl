# Include all schemes without rhs first. The rhs depends on the systems to define
# interactions between the different system types.
# Viscosity requires the open boundary system.
include("boundary/open_boundary/system.jl")
include("fluid/fluid.jl")
include("boundary/boundary.jl")
include("solid/total_lagrangian_sph/total_lagrangian_sph.jl")
include("solid/discrete_element_method/discrete_element_method.jl")
# Monaghan-Kajtar repulsive boundary particles require the `BoundarySPHSystem`
# and the `TotalLagrangianSPHSystem`.
include("boundary/monaghan_kajtar/monaghan_kajtar.jl")

# Include rhs for all schemes
include("fluid/weakly_compressible_sph/rhs.jl")
include("fluid/entropically_damped_sph/rhs.jl")
include("boundary/rhs.jl")
include("solid/total_lagrangian_sph/rhs.jl")
include("solid/discrete_element_method/rhs.jl")

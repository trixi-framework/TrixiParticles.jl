# Include all schemes without rhs first. The rhs depends on the systems to define
# interactions between the different system types.
include("fluid/fluid.jl")
include("boundary/boundary.jl")
include("solid/total_lagrangian_sph/total_lagrangian_sph.jl")

# Include rhs for all schemes
include("fluid/weakly_compressible_sph/rhs.jl")
include("boundary/rhs.jl")
include("solid/total_lagrangian_sph/rhs.jl")

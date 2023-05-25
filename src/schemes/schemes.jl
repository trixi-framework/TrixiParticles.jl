# Include all schemes first. The rhs depends on the systems to define
# interactions between the different system types.
include("fluid/weakly_compressible_sph/weakly_compressible_sph.jl")
include("boundary/boundary.jl")
include("solid/total_lagrangian_sph/total_lagrangian_sph.jl")

# include rhs
include("fluid/weakly_compressible_sph/rhs.jl")
include("boundary/rhs.jl")
include("solid/total_lagrangian_sph/rhs.jl")

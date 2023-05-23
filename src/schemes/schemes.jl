# Include all system types first. The rhs depends on the systems to define
# interactions between the different system types.
include("fluid/weakly_compressible_sph/system.jl")
include("boundary/system.jl")
include("solid/total_lagrangian_sph/system.jl")

# Include everything else
include("fluid/weakly_compressible_sph/weakly_compressible_sph.jl")
include("solid/total_lagrangian_sph/total_lagrangian_sph.jl")
include("boundary/boundary.jl")

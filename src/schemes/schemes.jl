# Include all container types first. The rhs depends on the containers to define
# interactions between the different container types.
include("fluid/weakly_compressible_sph/container.jl")
include("boundary/container.jl")
include("solid/total_lagrangian_sph/container.jl")

# Include everything else
include("fluid/weakly_compressible_sph/weakly_compressible_sph.jl")
include("solid/total_lagrangian_sph/total_lagrangian_sph.jl")
include("boundary/boundary.jl")

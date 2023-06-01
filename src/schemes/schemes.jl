# Include all schemes without rhs first. The rhs depends on the systems to define
# interactions between the different system types.
include("fluid/weakly_compressible_sph/weakly_compressible_sph.jl")
include("fluid/entropically_damped_sph/entropically_damped_sph.jl")
include("boundary/boundary.jl")
include("solid/total_lagrangian_sph/total_lagrangian_sph.jl")

# Include rhs for all schemes
include("fluid/weakly_compressible_sph/rhs.jl")
include("fluid/entropically_damped_sph/rhs.jl")
include("boundary/rhs.jl")
include("solid/total_lagrangian_sph/rhs.jl")

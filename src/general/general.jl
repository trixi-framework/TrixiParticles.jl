# Note that `abstract_system.jl` has already been included.
# `semidiscretization.jl` depends on the system types and has to be included later.
# `density_calculators.jl` needs to be included before `corrections.jl`.

# Generic fallback: ordered system pairs interact unless a semidiscretization-specific
# implementation overrides this hook.
@inline has_system_interaction(system, neighbor_system, semi) = true

include("density_calculators.jl")
include("corrections.jl")
include("smoothing_kernels.jl")
include("initial_condition.jl")
include("buffer.jl")
include("interpolation.jl")
include("custom_quantities.jl")
include("time_integration.jl")

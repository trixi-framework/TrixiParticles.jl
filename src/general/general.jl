# Note that `semidiscretization.jl` depends on the system types and has to be
# included later.
include("corrections.jl")
include("neighborhood_search.jl")
include("smoothing_kernels.jl")
include("initial_condition.jl")
include("system.jl")
include("density_calculators.jl")

# Note that `semidiscretization.jl` depends on the system types and has to be
# included later.
include("neighborhood_search.jl")
include("smoothing_kernels.jl")
include("density_calculators.jl")
include("initial_condition.jl")
include("container.jl")

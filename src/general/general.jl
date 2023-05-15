# Note that `semidiscretization.jl` depends on the container types and has to be
# included later.
include("neighborhood_search.jl")
include("smoothing_kernels.jl")
include("density_calculators.jl")
include("container.jl")

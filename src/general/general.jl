@inline function set_zero!(du)
    du .= zero(eltype(du))

    return du
end

# Note that `semidiscretization.jl` depends on the system types and has to be
# included later.
include("neighborhood_search.jl")
# needs to be before corrections.jl
include("density_calculators.jl")
include("corrections.jl")
include("smoothing_kernels.jl")
include("initial_condition.jl")
include("system.jl")

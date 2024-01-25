abstract type System{NDIMS} end

abstract type FluidSystem{NDIMS} <: System{NDIMS} end

abstract type SolidSystem{NDIMS} <: System{NDIMS} end

abstract type BoundarySystem{NDIMS} <: System{NDIMS} end

timer_name(::FluidSystem) = "fluid"

@inline function set_zero!(du)
    du .= zero(eltype(du))

    return du
end

# Note that `semidiscretization.jl` depends on the system types and has to be
# included later.
# `density_calculators.jl` needs to be included before `corrections.jl`.
include("density_calculators.jl")
include("corrections.jl")
include("smoothing_kernels.jl")
include("initial_condition.jl")
include("system.jl")
include("interpolation.jl")

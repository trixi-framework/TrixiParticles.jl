# Abstract supertype for all system types. We additionally store the type of the system's
# initial condition, which is `Nothing` when using KernelAbstractions.jl.
abstract type System{NDIMS, IC} end

# When using KernelAbstractions.jl, the initial condition has been replaced by `nothing`
GPUSystem = System{NDIMS, Nothing} where {NDIMS}

abstract type FluidSystem{NDIMS, IC} <: System{NDIMS, IC} end
timer_name(::FluidSystem) = "fluid"

abstract type SolidSystem{NDIMS, IC} <: System{NDIMS, IC} end
timer_name(::SolidSystem) = "solid"

abstract type BoundarySystem{NDIMS, IC} <: System{NDIMS, IC} end
timer_name(::BoundarySystem) = "boundary"

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
include("file_system.jl")
include("custom_quantities.jl")
include("neighborhood_search.jl")

abstract type System{NDIMS} end

# WCSPH EDAC
abstract type FluidSystem{NDIMS} <: System{NDIMS} end

# TLSPH
abstract type SolidSystem{NDIMS} <: System{NDIMS} end

# all Boundary Condition Systems
abstract type BoundarySystem{NDIMS} <: System{NDIMS} end

timer_name(::FluidSystem) = "fluid"

@inline function set_zero!(du)
    du .= zero(eltype(du))

    return du
end

@inline function invert(inverse, A, particle, system)
    A_inv = inv(A)
    @inbounds for j in 1:ndims(system), i in 1:ndims(system)
        inverse[i, j, particle] = A_inv[i, j]
    end
end

@inline function pseudo_invert(inverse, A, particle, system)
    A_inv = pinv(A)
    @inbounds for j in 1:ndims(system), i in 1:ndims(system)
        inverse[i, j, particle] = A_inv[i, j]
    end
end

# Note that `semidiscretization.jl` depends on the system types and has to be
# included later.
# `density_calculators.jl` needs to be included before `corrections.jl`.
include("density_calculators.jl")
include("corrections.jl")
include("smoothing_kernels.jl")
include("initial_condition.jl")
include("system.jl")

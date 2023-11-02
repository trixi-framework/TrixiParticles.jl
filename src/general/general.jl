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

@inline function invert_factorization(inverse, A, particle, system)
    A_factored = lu(A)
    A_inv = similar(A)

    n = ndims(system)
    y = similar(A_inv[:, 1])

    # Inverting by solving linear system for each column of the identity
    @inbounds for j in 1:n
        e_j = zeros(eltype(A), n)
        e_j[j] = 1.0
        y .= A_factored.L \ e_j
        A_inv[:, j] .= A_factored.U \ y
    end
    @inbounds for j in 1:n, i in 1:n
        inverse[i, j, particle] = A_inv[i, j]
    end
end

struct SimulationDiverged <: Exception
    msg::AbstractString
end

struct ModelError <: Exception
    msg::AbstractString
end

# Note that `semidiscretization.jl` depends on the system types and has to be
# included later.
# `density_calculators.jl` needs to be included before `corrections.jl`.
include("density_calculators.jl")
include("corrections.jl")
include("smoothing_kernels.jl")
include("initial_condition.jl")
include("system.jl")

# Adapt.jl provides a function `adapt(to, x)`, which adapts a value `x` to `to`.
# In practice, this means that we can use `adapt(CuArray, system)` to adapt a system to
# the `CuArray` type.
# What this does is that it converts all `Array`s inside this system to `CuArray`s,
# therefore copying them to the GPU.
# In order to run a simulation on a GPU, we want to call `adapt(T, systems)` to adapt
# all systems in the `Semidiscretization` to the GPU array type `T` (e.g. `CuArray`).
#
# `Adapt.@adapt_structure` automatically generates the `adapt` function for our custom types.
Adapt.@adapt_structure InitialCondition
Adapt.@adapt_structure WeaklyCompressibleSPHSystem
Adapt.@adapt_structure DensityDiffusionAntuono
Adapt.@adapt_structure EntropicallyDampedSPHSystem
Adapt.@adapt_structure WallBoundarySystem
Adapt.@adapt_structure BoundaryModelDummyParticles
Adapt.@adapt_structure BoundaryModelMonaghanKajtar
Adapt.@adapt_structure PrescribedMotion
Adapt.@adapt_structure TotalLagrangianSPHSystem
Adapt.@adapt_structure BoundaryZone
Adapt.@adapt_structure SystemBuffer
Adapt.@adapt_structure OpenBoundarySystem
Adapt.@adapt_structure DEMSystem
Adapt.@adapt_structure BoundaryDEMSystem
Adapt.@adapt_structure RCRWindkesselModel

# This makes `@threaded semi for ...` use `semi.parallelization_backend` for parallelization
@inline function PointNeighbors.parallel_foreach(f, iterator, semi::Semidiscretization)
    PointNeighbors.parallel_foreach(f, iterator, semi.parallelization_backend)
end

function allocate(backend::KernelAbstractions.GPU, ELTYPE, size)
    return KernelAbstractions.allocate(backend, ELTYPE, size)
end

function allocate(backend, ELTYPE, size)
    return Array{ELTYPE, length(size)}(undef, size)
end

# This function checks if we can use aligned `SVector` or `SMatrix` loads for the given
# array `A` and the size of the vector we want to load `n`.
#
# Note that it is not checked if the size `n` is legal (a power of 2).
# If the size `n` is not legal, the aligned loads will fall back to regular loads.
# This is because a regular load is expected for a problem that doesn't support aligned
# loads (e.g. 3D matrices of 3x3), whereas an illegal alignment is unexpected and should
# throw an error instead of silently (and non-deterministically) falling back to slower
# regular loads.
function can_use_aligned_load(A, n)
    # Check if the beginning of `A` is aligned to the size of the vector we want to load.
    is_aligned = UInt(pointer(A)) % (n * sizeof(eltype(A))) == 0
    # Check if the stride of `A` in the last dimension is equal to `n`,
    # which means that there is no padding between the vectors we want to load.
    has_stride = stride(A, ndims(A)) == n

    return is_aligned && has_stride
end

# For N = 2 and N = 4, use the aligned vector loads.
@propagate_inbounds function extract_svector_aligned(A::AbstractMatrix,
                                                     val_n::Union{Val{2}, Val{4}}, i)
    return _extract_svector_aligned(A, val_n, i)
end

# For other N, fall back to the regular `extract_svector`.
@propagate_inbounds function extract_svector_aligned(A, val_n, i)
    return extract_svector(A, val_n, i)
end

# This function only works for N = 2 and N = 4, because it requires a stride of 2^n.
@inline function _extract_svector_aligned(A::AbstractMatrix{T}, ::Val{N}, i) where {T, N}
    @boundscheck checkbounds(A, 1:N, i)

    # This function assumes alignment of the data, which means that the columns of `A`
    # have exactly the size `N` and no padding between them.
    @boundscheck @assert stride(A, 2) == N

    vec = SIMD.vloada(SIMD.Vec{N, eltype(A)}, pointer(A, N * (i - 1) + 1))

    return SVector{N}(Tuple(vec))
end

# For general N, fall back to the regular `extract_smatrix`.
@propagate_inbounds function extract_matrix_aligned(A, val_n, i)
    return extract_smatrix(A, val_n, i)
end

# This function only works for 2x2 matrices, because it requires a stride of 2^n.
@inline function extract_smatrix_aligned(A::AbstractArray{T, 3}, ::Val{2}, i) where {T}
    @boundscheck checkbounds(A, 2, 2, i)

    # This function assumes alignment of the data, which means that the first two of `A`
    # have exactly the size `2` and no padding between them.
    @boundscheck @assert stride(A, 3) == 4

    vec = SIMD.vloada(SIMD.Vec{4, T}, pointer(A, 4 * (i - 1) + 1))

    return SMatrix{2, 2}(Tuple(vec))
end

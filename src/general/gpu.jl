# Adapt.jl provides a function `adapt(to, x)`, which adapts a value `x` to `to`.
# In practice, this means that we can use `adapt(CuArray, system)` to adapt a system to
# the `CuArray` type.
# What this does is that it converts all `Array`s inside this system to `CuArray`s,
# therefore copying them to the GPU.
# In order to run a simulation on a GPU, we want to call `adapt(T, semi)` to adapt the
# `Semidiscretization` `semi` to the GPU array type `T` (e.g. `CuArray`).
#
# `Adapt.@adapt_structure` automatically generates the `adapt` function for our custom types.
Adapt.@adapt_structure Semidiscretization
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

KernelAbstractions.get_backend(::PtrArray) = KernelAbstractions.CPU()
function KernelAbstractions.get_backend(system::AbstractSystem)
    KernelAbstractions.get_backend(system.mass)
end

function KernelAbstractions.get_backend(system::WallBoundarySystem)
    KernelAbstractions.get_backend(system.coordinates)
end

# This makes `@threaded semi for ...` use `semi.parallelization_backend` for parallelization
@inline function PointNeighbors.parallel_foreach(f, iterator, semi::Semidiscretization)
    PointNeighbors.parallel_foreach(f, iterator, semi.parallelization_backend)
end

function allocate(backend::KernelAbstractions.Backend, ELTYPE, size)
    return KernelAbstractions.allocate(backend, ELTYPE, size)
end

function allocate(backend, ELTYPE, size)
    return Array{ELTYPE, length(size)}(undef, size)
end

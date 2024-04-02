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
Adapt.@adapt_structure WeaklyCompressibleSPHSystem
Adapt.@adapt_structure DensityDiffusionAntuono
Adapt.@adapt_structure BoundarySPHSystem
Adapt.@adapt_structure BoundaryModelDummyParticles
Adapt.@adapt_structure BoundaryModelMonaghanKajtar

# The initial conditions are only used for initialization, which happens before `adapt`ing
# the semidiscretization, so we don't need to store `InitialCondition`s on the GPU.
# To save precious GPU memory, we replace initial conditions by `nothing`.
function Adapt.adapt_structure(to, ic::InitialCondition)
    return nothing
end

# `adapt(CuArray, ::SVector)::SVector`, but `adapt(Array, ::SVector)::Vector`.
# We don't want to change the type of the `SVector` here.
function Adapt.adapt_structure(to::typeof(Array), svector::SVector)
    return svector
end

# `adapt(CuArray, ::UnitRange)::UnitRange`, but `adapt(Array, ::UnitRange)::Vector`.
# We don't want to change the type of the `UnitRange` here.
function Adapt.adapt_structure(to::typeof(Array), range::UnitRange)
    return range
end

KernelAbstractions.get_backend(::PtrArray) = KernelAbstractions.CPU()

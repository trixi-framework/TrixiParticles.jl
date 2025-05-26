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
Adapt.@adapt_structure BoundarySPHSystem
Adapt.@adapt_structure BoundaryModelDummyParticles
Adapt.@adapt_structure BoundaryModelMonaghanKajtar
Adapt.@adapt_structure BoundaryMovement
Adapt.@adapt_structure TotalLagrangianSPHSystem
Adapt.@adapt_structure BoundaryZone
Adapt.@adapt_structure SystemBuffer

function Adapt.adapt_structure(to, system::OpenBoundarySPHSystem)
    return OpenBoundarySPHSystem(Adapt.adapt(to, system.boundary_model),
                                 Adapt.adapt(to, system.initial_condition),
                                 nothing,  # Do not adapt `fluid_system``
                                 Adapt.adapt(to, system.fluid_system_index),
                                 Adapt.adapt(to, system.smoothing_length),
                                 Adapt.adapt(to, system.mass),
                                 Adapt.adapt(to, system.density),
                                 Adapt.adapt(to, system.volume),
                                 Adapt.adapt(to, system.pressure),
                                 Adapt.adapt(to, system.boundary_zone),
                                 Adapt.adapt(to, system.reference_velocity),
                                 Adapt.adapt(to, system.reference_pressure),
                                 Adapt.adapt(to, system.reference_density),
                                 Adapt.adapt(to, system.buffer),
                                 Adapt.adapt(to, system.update_callback_used),
                                 Adapt.adapt(to, system.cache))
end

KernelAbstractions.get_backend(::PtrArray) = KernelAbstractions.CPU()
KernelAbstractions.get_backend(system::System) = KernelAbstractions.get_backend(system.mass)

function KernelAbstractions.get_backend(system::BoundarySPHSystem)
    KernelAbstractions.get_backend(system.coordinates)
end

# This makes `@threaded semi for ...` use `semi.parallelization_backend` for parallelization
@inline function PointNeighbors.parallel_foreach(f, iterator, semi::Semidiscretization)
    PointNeighbors.parallel_foreach(f, iterator, semi.parallelization_backend)
end

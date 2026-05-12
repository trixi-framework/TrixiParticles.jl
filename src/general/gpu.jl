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

function transfer2cpu(a::AbstractGPUArray)
    return Adapt.adapt(Array, a)
end

function transfer2cpu(a)
    return a
end

function transfer2cpu(v_, u_)
    v = transfer2cpu(v_)
    u = transfer2cpu(u_)

    return v, u
end

function transfer2cpu(semi::Semidiscretization)
    # First move all systems and neighborhood searches to the CPU
    systems = Adapt.adapt(Array, semi.systems)
    neighborhood_searches = Adapt.adapt(Array, semi.neighborhood_searches)

    semi_ = @set semi.systems = systems
    semi__ = @set semi_.neighborhood_searches = neighborhood_searches

    # Now, set the parallelization backend to `PolyesterBackend` to make sure that
    # `@threaded` loops still work as expected with this semidiscretization.
    return @set semi__.parallelization_backend = PolyesterBackend()
end

function transfer2cpu(v_::AbstractGPUArray, u_, semi_)
    semi = transfer2cpu(semi_)
    v, u = transfer2cpu(v_, u_)

    return v, u, semi
end

function transfer2cpu(v_, u_, semi_)
    return v_, u_, semi_
end

function transfer2cpu(v_::AbstractGPUArray, u_, system_, semi_)
    v, u, semi = transfer2cpu(v_, u_, semi_)
    system_index = system_indices(system_, semi_)
    system = semi.systems[system_index]

    return v, u, system, semi
end

function transfer2cpu(v_, u_, system_, semi_)
    return v_, u_, system_, semi_
end

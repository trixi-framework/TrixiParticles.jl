# Same as `foreach`, but it optimizes away for small input tuples
@inline function foreach_noalloc(func, collection)
    element = first(collection)
    remaining_collection = Base.tail(collection)

    func(element)

    # Process remaining collection
    foreach_noalloc(func, remaining_collection)
end

@inline foreach_noalloc(func, collection::Tuple{}) = nothing

# Same as `foreach(enumerate(something))`, but without allocations.
# Note that compile times may increase if this is used with big tuples.
@inline foreach_enumerate(func, collection) = foreach_enumerate(func, collection, 1)
@inline foreach_enumerate(func, collection::Tuple{}, index) = nothing

@inline function foreach_enumerate(func, collection, index)
    element = first(collection)
    remaining_collection = Base.tail(collection)

    @inline func((index, element))

    # Process remaining collection
    foreach_enumerate(func, remaining_collection, index + 1)
end

# Returns `functions[index](args...)`, but in a type-stable way for a heterogeneous tuple `functions`
@inline function apply_ith_function(functions, index, args...)
    if index == 1
        # Found the function to apply, apply it and return
        return first(functions)(args...)
    end

    # Process remaining functions
    apply_ith_function(Base.tail(functions), index - 1, args...)
end

# Print informative message at startup
function print_startup_message()
    s = """

        ████████╗██████╗ ██╗██╗  ██╗██╗██████╗  █████╗ ██████╗ ████████╗██╗ ██████╗██╗     ███████╗███████╗
        ╚══██╔══╝██╔══██╗██║╚██╗██╔╝██║██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██║██╔════╝██║     ██╔════╝██╔════╝
           ██║   ██████╔╝██║ ╚███╔╝ ██║██████╔╝███████║██████╔╝   ██║   ██║██║     ██║     █████╗  ███████╗
           ██║   ██╔══██╗██║ ██╔██╗ ██║██╔═══╝ ██╔══██║██╔══██╗   ██║   ██║██║     ██║     ██╔══╝  ╚════██║
           ██║   ██║  ██║██║██╔╝ ██╗██║██║     ██║  ██║██║  ██║   ██║   ██║╚██████╗███████╗███████╗███████║
           ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝╚══════╝╚══════╝╚══════╝

        """
    println(s)
end

@doc raw"""
    examples_dir()

Return the directory where the example files provided with TrixiParticles.jl are located. If TrixiParticles is
installed as a regular package (with `]add TrixiParticles`), these files are read-only and should *not* be
modified. To find out which files are available, use, e.g., `readdir`.

Copied from [Trixi.jl](https://github.com/trixi-framework/Trixi.jl).

# Examples
```jldoctest; output = false, filter = r"\d+-element Vector.*"s
readdir(examples_dir())

# output
7-element Vector{String}:
 [...] (the rest is ignored by filter condition)
```
"""
examples_dir() = pkgdir(TrixiParticles, "examples")

"""
    validation_dir()

Return the directory where the validation files provided with TrixiParticles.jl are located. If TrixiParticles is
installed as a regular package (with `]add TrixiParticles`), these files are read-only and should *not* be
modified. To find out which files are available, use, e.g., `readdir`.

Copied from [Trixi.jl](https://github.com/trixi-framework/Trixi.jl).

# Examples
```@example
readdir(validation_dir())
```
"""
validation_dir() = pkgdir(TrixiParticles, "validation")

"""
    @autoinfiltrate
    @autoinfiltrate condition::Bool

Invoke the `@infiltrate` macro of the package Infiltrator.jl to create a breakpoint for ad-hoc
interactive debugging in the REPL. If the optional argument `condition` is given, the breakpoint is
only enabled if `condition` evaluates to `true`.

As opposed to using `Infiltrator.@infiltrate` directly, this macro does not require Infiltrator.jl
to be added as a dependency to TrixiParticles.jl. As a bonus, the macro will also attempt to load
the Infiltrator module if it has not yet been loaded manually.

Note: For this macro to work, the Infiltrator.jl package needs to be installed in your current Julia
environment stack.

See also: [Infiltrator.jl](https://github.com/JuliaDebug/Infiltrator.jl)

!!! warning "Internal use only"
    Please note that this macro is intended for internal use only. It is *not* part of the public
    API of TrixiParticles.jl, and it thus can altered (or be removed) at any time without it being
    considered a breaking change.
"""
macro autoinfiltrate(condition=true)
    pkgid = Base.PkgId(Base.UUID("5903a43b-9cc3-4c30-8d17-598619ec4e9b"), "Infiltrator")
    if !haskey(Base.loaded_modules, pkgid)
        try
            Base.eval(Main, :(using Infiltrator))
        catch err
            @error "Cannot load Infiltrator.jl. Make sure it is included in your environment stack."
        end
    end
    i = get(Base.loaded_modules, pkgid, nothing)
    lnn = LineNumberNode(__source__.line, __source__.file)

    if i === nothing
        return Expr(:macrocall,
                    Symbol("@warn"),
                    lnn,
                    "Could not load Infiltrator.")
    end

    return Expr(:macrocall,
                Expr(:., i, QuoteNode(Symbol("@infiltrate"))),
                lnn,
                esc(condition))
end

function type2string(type)
    return string(nameof(typeof(type)))
end

function type2string(type::Function)
    return string(nameof(type))
end

function compute_git_hash()
    pkg_directory = pkgdir(@__MODULE__)
    git_directory = joinpath(pkg_directory, ".git")

    # Check if the .git directory exists
    if !isdir(git_directory)
        return "UnknownVersion"
    end

    try
        git_cmd = Cmd(`git describe --tags --always --dirty`,
                      dir=pkg_directory)
        return string(readchomp(git_cmd))
    catch e
        return "UnknownVersion"
    end
end

# This data type wraps regular arrays and redefines broadcasting and common operations
# like `fill!` and `copyto!` to use multithreading with `@threaded`.
# See https://github.com/trixi-framework/TrixiParticles.jl/pull/722 for more details
# and benchmarks.
struct ThreadedBroadcastArray{T, N, A <: AbstractArray{T, N}, P} <: AbstractArray{T, N}
    array::A
    parallelization_backend::P

    function ThreadedBroadcastArray(array::AbstractArray{T, N};
                                    parallelization_backend=default_backend(array)) where {T,
                                                                                           N}
        new{T, N, typeof(array), typeof(parallelization_backend)}(array,
                                                                  parallelization_backend)
    end
end

Base.parent(A::ThreadedBroadcastArray) = A.array
Base.pointer(A::ThreadedBroadcastArray) = pointer(parent(A))
Base.size(A::ThreadedBroadcastArray) = size(parent(A))
Base.IndexStyle(::Type{<:ThreadedBroadcastArray}) = IndexLinear()

function Base.similar(A::ThreadedBroadcastArray, ::Type{T}) where {T}
    return ThreadedBroadcastArray(similar(A.array, T);
                                  parallelization_backend=A.parallelization_backend)
end

function Base.convert(::Type{ThreadedBroadcastArray{T, N, A, P}},
                      a::AbstractArray) where {T, N, A, P}
    if a isa ThreadedBroadcastArray{T, N, A, P}
        return a
    end

    # TODO we only have the type `P` here and just assume that we can do `P()`
    return ThreadedBroadcastArray(convert(A, a), parallelization_backend=P())
end

Base.@propagate_inbounds function Base.getindex(A::ThreadedBroadcastArray, i...)
    return getindex(A.array, i...)
end

Base.@propagate_inbounds function Base.setindex!(A::ThreadedBroadcastArray, x, i...)
    setindex!(A.array, x, i...)
    return A
end

# For things like `A .= 0` where `A` is a `ThreadedBroadcastArray`
function Base.fill!(A::ThreadedBroadcastArray{T}, x) where {T}
    xT = x isa T ? x : convert(T, x)::T
    @threaded A.parallelization_backend for i in eachindex(A.array)
        @inbounds A.array[i] = xT
    end

    return A
end

# Based on
# copyto_unaliased!(deststyle::IndexStyle, dest::AbstractArray, srcstyle::IndexStyle, src::AbstractArray)
# defined in base/abstractarray.jl.
function Base.copyto!(dest::ThreadedBroadcastArray, src::AbstractArray)
    if eachindex(dest) == eachindex(src)
        # Shared-iterator implementation
        @threaded dest.parallelization_backend for I in eachindex(dest)
            @inbounds dest.array[I] = src[I]
        end
    else
        # Dual-iterator implementation
        @threaded dest.parallelization_backend for (Idest, Isrc) in zip(eachindex(dest),
                                                       eachindex(src))
            @inbounds dest.array[Idest] = src[Isrc]
        end
    end

    return dest
end

# Broadcasting style for `ThreadedBroadcastArray`.
struct ThreadedBroadcastStyle{P} <: Broadcast.AbstractArrayStyle{1} end

function Broadcast.BroadcastStyle(::Type{ThreadedBroadcastArray{T, N, A, P}}) where {T, N,
                                                                                     A, P}
    return ThreadedBroadcastStyle{P}()
end

# The threaded broadcast style wins over any other array style.
# For things like `A .+ B` where `A` is a `ThreadedBroadcastArray` and `B` is a
# `RecursiveArrayTools.ArrayPartition`.
#
# https://docs.julialang.org/en/v1/manual/interfaces/
# "It is worth noting that you do not need to (and should not) define both argument orders
# of this call;
# defining one is sufficient no matter what order the user supplies the arguments in."
function Broadcast.BroadcastStyle(s1::ThreadedBroadcastStyle,
                                  ::Broadcast.AbstractArrayStyle)
    return s1
end

# To avoid ambiguity with the function above
function Broadcast.BroadcastStyle(s1::ThreadedBroadcastStyle,
                                  ::Broadcast.DefaultArrayStyle)
    return s1
end

# Based on copyto!(dest::AbstractArray, bc::Broadcasted{Nothing})
# defined in base/broadcast.jl.
# For things like `A .= B .+ C` where `A` is a `ThreadedBroadcastArray`.
function Broadcast.copyto!(dest::ThreadedBroadcastArray,
                           bc::Broadcast.Broadcasted{Nothing})
    # Check bounds
    axes(dest.array) == axes(bc) || Broadcast.throwdm(axes(dest.array), axes(bc))

    bc_ = Base.Broadcast.preprocess(dest.array, bc)

    @threaded dest.parallelization_backend for i in eachindex(bc_)
        @inbounds dest.array[i] = bc_[i]
    end
    return dest
end

# For things like `C = A .+ B` where `A` or `B` is a `ThreadedBroadcastArray`.
# `C` will be allocated with this function.
function Base.similar(::Broadcast.Broadcasted{ThreadedBroadcastStyle{P}},
                      ::Type{T}, dims) where {T, P}
    # TODO we only have the type `P` here and just assume that we can do `P()`
    return ThreadedBroadcastArray(similar(Array{T}, dims), parallelization_backend=P())
end

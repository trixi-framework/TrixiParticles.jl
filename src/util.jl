# Same as `foreach`, but it optimizes away for small input tuples
@inline function foreach_noalloc(func, collection)
    element = first(collection)
    remaining_collection = Base.tail(collection)

    func(element)

    # Process remaining collection
    foreach_noalloc(func, remaining_collection)
end

@inline foreach_noalloc(func, collection::Tuple{}) = nothing

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
        git_cmd = Cmd(`git describe --tags --always --first-parent --dirty`,
                      dir=pkg_directory)
        return string(readchomp(git_cmd))
    catch e
        return "UnknownVersion"
    end
end

struct ThreadedBroadcastArray{T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
    array::A

    function ThreadedBroadcastArray(array::AbstractArray{T, N}) where {T, N}
        new{T, N, typeof(array)}(array)
    end
end

Base.parent(A::ThreadedBroadcastArray) = A.array
Base.pointer(A::ThreadedBroadcastArray) = pointer(parent(A))
Base.size(A::ThreadedBroadcastArray) = size(parent(A))

function Base.similar(A::ThreadedBroadcastArray, ::Type{T}) where {T}
    return ThreadedBroadcastArray(similar(A.array, T))
end

Base.@propagate_inbounds function Base.getindex(A::ThreadedBroadcastArray, i...)
    return getindex(A.array, i...)
end

Base.@propagate_inbounds function Base.setindex!(A::ThreadedBroadcastArray, x...)
    setindex!(A.array, x...)
    return A
end

function Base.fill!(A::ThreadedBroadcastArray{T}, x) where {T}
    xT = x isa T ? x : convert(T, x)::T
    @threaded A.array for i in eachindex(A.array)
        @inbounds A.array[i] = xT
    end

    return A
end

function Base.copyto!(dest::ThreadedBroadcastArray, src::AbstractArray)
    if eachindex(dest) == eachindex(src)
        # Shared-iterator implementation
        @threaded dest.array for I in eachindex(dest)
            @inbounds dest.array[I] = src[I]
        end
    else
        # Dual-iterator implementation
        @threaded dest.array for (Idest, Isrc) in zip(eachindex(dest), eachindex(src))
            @inbounds dest.array[Idest] = src[Isrc]
        end
    end

    return dest
end

function Broadcast.BroadcastStyle(::Type{ThreadedBroadcastArray{T, N, A}}) where {T, N, A}
    return Broadcast.ArrayStyle{ThreadedBroadcastArray}()
end

function Broadcast.copyto!(dest::ThreadedBroadcastArray,
                           bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{ThreadedBroadcastArray}})
    # Check bounds
    axes(dest.array) == axes(bc) || Broadcast.throwdm(axes(dest.array), axes(bc))

    @threaded dest.array for i in eachindex(dest.array)
        @inbounds dest.array[i] = bc[i]
    end
    return dest
end

# function Base.copyto!(dest::ThreadedBroadcastArray, indices1::CartesianIndices, src::AbstractArray, indices2::CartesianIndices)
#     copyto!(dest.array, indices1, src, indices2)
#     return dest
# end

# function Base.copyto!(dest::AbstractArray, src::ThreadedBroadcastArray)
#     copyto!(dest, src.array)
#     return dest
# end

# function Base.copyto!(dest::ThreadedBroadcastArray, src::ThreadedBroadcastArray)
#     # TODO check bounds
#     @threaded dest for i in eachindex(dest.array)
#         @inbounds dest.array[i] = src.array[i]
#     end

#     return dest
# end

# Base.view(m::ThreadedBroadcastArray, i) = ThreadedBroadcastArray(view(m.array, i))
# Base.reshape(m::ThreadedBroadcastArray, dims::Base.Dims) = ThreadedBroadcastArray(reshape(m.array, dims))

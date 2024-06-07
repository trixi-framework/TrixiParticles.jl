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

"""
    @threaded for ... end

Semantically the same as `Threads.@threads` when iterating over a `AbstractUnitRange`
but without guarantee that the underlying implementation uses `Threads.@threads`
or works for more general `for` loops.
In particular, there may be an additional check whether only one thread is used
to reduce the overhead of serial execution or the underlying threading capabilities
might be provided by other packages such as [Polyester.jl](https://github.com/JuliaSIMD/Polyester.jl).

!!! warn
    This macro does not necessarily work for general `for` loops. For example,
    it does not necessarily support general iterables such as `eachline(filename)`.

Some discussion can be found at
[https://discourse.julialang.org/t/overhead-of-threads-threads/53964](https://discourse.julialang.org/t/overhead-of-threads-threads/53964)
and
[https://discourse.julialang.org/t/threads-threads-with-one-thread-how-to-remove-the-overhead/58435](https://discourse.julialang.org/t/threads-threads-with-one-thread-how-to-remove-the-overhead/58435).

Copied from [Trixi.jl](https://github.com/trixi-framework/Trixi.jl).
"""
macro threaded(expr)
    # Use `esc(quote ... end)` for nested macro calls as suggested in
    # https://github.com/JuliaLang/julia/issues/23221
    #
    # The following code is a simple version using only `Threads.@threads` from the
    # standard library with an additional check whether only a single thread is used
    # to reduce some overhead (and allocations) for serial execution.
    #
    # return esc(quote
    #   let
    #     if Threads.nthreads() == 1
    #       $(expr)
    #     else
    #       Threads.@threads $(expr)
    #     end
    #   end
    # end)
    #
    # However, the code below using `@batch` from Polyester.jl is more efficient,
    # since this packages provides threads with less overhead. Since it is written
    # by Chris Elrod, the author of LoopVectorization.jl, we expect this package
    # to provide the most efficient and useful implementation of threads (as we use
    # them) available in Julia.
    # !!! danger "Heisenbug"
    #     Look at the comments for `wrap_array` when considering to change this macro.

    return esc(quote
                   TrixiParticles.@batch $(expr)
               end)
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

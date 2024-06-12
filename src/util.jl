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

# Enable debug timings `@trixi_timeit timer() "name" stuff...`.
# This allows us to disable timings completely by executing
# `TimerOutputs.disable_debug_timings(TrixiParticles)`
# and to enable them again by executing
# `TimerOutputs.enable_debug_timings(TrixiParticles)`
timeit_debug_enabled() = true

# Store main timer for global timing of functions
const main_timer = TimerOutput()

# Always call timer() to hide implementation details
timer() = main_timer

#     @trixi_timeit timer() "some label" expression
#
# Basically the same as a special case of `@timeit_debug` from
# [TimerOutputs.jl](https://github.com/KristofferC/TimerOutputs.jl),
# but without `try ... finally ... end` block. Thus, it's not exception-safe,
# but it also avoids some related performance problems. Since we do not use
# exception handling in TrixiParticles, that's not really an issue.
#
# Copied from [Trixi.jl](https://github.com/trixi-framework/Trixi.jl).
macro trixi_timeit(timer_output, label, expr)
    timeit_block = quote
        if timeit_debug_enabled()
            local to = $(esc(timer_output))
            local enabled = to.enabled
            if enabled
                local accumulated_data = $(TimerOutputs.push!)(to, $(esc(label)))
            end
            local b0 = $(TimerOutputs.gc_bytes)()
            local t0 = $(TimerOutputs.time_ns)()
        end
        local val = $(esc(expr))
        if timeit_debug_enabled() && enabled
            $(TimerOutputs.do_accumulate!)(accumulated_data, t0, b0)
            $(TimerOutputs.pop!)(to)
        end
        val
    end
end

"""
    @threaded system for ... end

Run either a threaded CPU loop or launch a kernel on the GPU, depending on the type of `system`.
Semantically the same as `Threads.@threads` when iterating over a `AbstractUnitRange`
but without guarantee that the underlying implementation uses `Threads.@threads`
or works for more general `for` loops.
The second argument must either be a particle system or an array from which can be derived
if the loop has to be run threaded on the CPU or launched as a kernel on the GPU.

In particular, the underlying threading capabilities might be provided by other packages
such as [Polyester.jl](https://github.com/JuliaSIMD/Polyester.jl).

!!! warn
    This macro does not necessarily work for general `for` loops. For example,
    it does not necessarily support general iterables such as `eachline(filename)`.
"""
macro threaded(system, expr)
    # Reverse-engineer the for loop.
    # `expr.args[1]` is the head of the for loop, like `i = eachindex(x)`.
    # So, `expr.args[1].args[2]` is the iterator `eachindex(x)`
    # and `expr.args[1].args[1]` is the loop variable `i`.
    iterator = expr.args[1].args[2]
    i = expr.args[1].args[1]
    inner_loop = expr.args[2]

    # Assemble the for loop again as a call to `parallel_foreach`, using `$i` to use the
    # same loop variable as used in the for loop.
    return esc(quote
                   TrixiParticles.parallel_foreach($iterator, $system) do $i
                       $inner_loop
                   end
               end)
end

# Use `Polyester.@batch` for low-overhead threading
@inline function parallel_foreach(f, iterator, system)
    Polyester.@batch for i in iterator
        @inline f(i)
    end
end

# On GPUs, execute `f` inside a GPU kernel with KernelAbstractions.jl
@inline function parallel_foreach(f, iterator, system::Union{GPUSystem, AbstractGPUArray})
    # On the GPU, we can only loop over `1:N`. Therefore, we loop over `1:length(iterator)`
    # and index with `iterator[eachindex(iterator)[i]]`.
    # Note that this only works with vector-like iterators that support arbitrary indexing.
    indices = eachindex(iterator)
    ndrange = length(indices)

    # Skip empty loops
    ndrange == 0 && return

    backend = KernelAbstractions.get_backend(system)

    # Call the generic kernel that is defined below, which only calls a function with
    # the global GPU index.
    generic_kernel(backend)(ndrange=ndrange) do i
        @inline f(iterator[indices[i]])
    end

    KernelAbstractions.synchronize(backend)
end

@kernel function generic_kernel(f)
    i = @index(Global)
    @inline f(i)
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

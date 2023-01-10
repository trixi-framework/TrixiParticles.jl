# Same as `foreach(enumerate(something))`, but without allocations.
#
# Note that compile times may increase if this is used with big tuples.
@inline foreach_enumerate(func, collection) = foreach_enumerate(func, collection, 1)
@inline foreach_enumerate(func, collection::Tuple{}, index) = nothing

@inline function foreach_enumerate(func, collection, index)
    element = first(collection)
    remaining_collection = Base.tail(collection)

    func((index, element))

    # Process remaining collection
    foreach_enumerate(func, remaining_collection, index + 1)
end


# Print informative message at startup
function print_startup_message(dic_info)
    s = """

        ██████╗ ██╗██╗  ██╗██╗███████╗
        ██╔══██╗██║╚██╗██╔╝██║██╔════╝
        ██████╔╝██║ ╚███╔╝ ██║█████╗
        ██╔═══╝ ██║ ██╔██╗ ██║██╔══╝
        ██║     ██║██╔╝ ██╗██║███████╗
        ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚══════╝
        """
    println(s)

end

# Enable debug timings `@pixie_timeit timer() "name" stuff...`.
# This allows us to disable timings completely by executing
# `TimerOutputs.disable_debug_timings(Pixie)`
# and to enable them again by executing
# `TimerOutputs.enable_debug_timings(Pixie)`
timeit_debug_enabled() = true

# Store main timer for global timing of functions
const main_timer = TimerOutput()

# Always call timer() to hide implementation details
timer() = main_timer

#     @pixie_timeit timer() "some label" expression
#
# Basically the same as a special case of `@timeit_debug` from
# [TimerOutputs.jl](https://github.com/KristofferC/TimerOutputs.jl),
# but without `try ... finally ... end` block. Thus, it's not exception-safe,
# but it also avoids some related performance problems. Since we do not use
# exception handling in Pixie, that's not really an issue.
#
# Copied from [Trixi.jl](https://github.com/trixi-framework/Trixi.jl).
macro pixie_timeit(timer_output, label, expr)
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

    return esc(quote Pixie.@batch $(expr) end)
end


"""
    examples_dir()

Return the directory where the example files provided with Pixie.jl are located. If Pixie is
installed as a regular package (with `]add Pixie`), these files are read-only and should *not* be
modified. To find out which files are available, use, e.g., `readdir`.

Copied from [Trixi.jl](https://github.com/trixi-framework/Trixi.jl).

# Examples
```@example
readdir(examples_dir())
```
"""
examples_dir() = joinpath(pathof(Pixie) |> dirname |> dirname, "examples")


# Note: We can't call the method below `Pixie.include` since that is created automatically
# inside `module Pixie` to `include` source files and evaluate them within the global scope
# of `Pixie`. However, users will want to evaluate in the global scope of `Main` or something
# similar to manage dependencies on their own.
"""
    pixie_include([mod::Module=Main,] example::AbstractString; kwargs...)

`include` the file `example` and evaluate its content in the global scope of module `mod`.
You can override specific assignments in `example` by supplying keyword arguments.
It's basic purpose is to make it easier to modify some parameters while running Pixie from the
REPL. Additionally, this is used in tests to reduce the computational burden for CI while still
providing examples with sensible default values for users.

Before replacing assignments in `example`, the keyword argument `maxiters` is inserted
into calls to `solve` and `Pixie.solve` with it's default value used in the SciML ecosystem
for ODEs, see https://diffeq.sciml.ai/stable/basics/common_solver_opts/#Miscellaneous.

Copied from [Trixi.jl](https://github.com/trixi-framework/Trixi.jl).

# Examples

```jldoctest
julia> redirect_stdout(devnull) do
           pixie_include(@__MODULE__, joinpath(examples_dir(), "dam_break_2d.jl"),
                         tspan=(0.0, 0.1))
           sol.t[end]
       end
0.1
```
"""
function pixie_include(mod::Module, elixir::AbstractString; kwargs...)
    Base.include(ex -> replace_assignments(insert_maxiters(ex); kwargs...), mod, elixir)
end

pixie_include(elixir::AbstractString; kwargs...) = pixie_include(Main, elixir; kwargs...)

# Helper methods used in the functions defined above, also copied from Trixi.jl

# Apply the function `f` to `expr` and all sub-expressions recursively.
walkexpr(f, expr::Expr) = f(Expr(expr.head, (walkexpr(f, arg) for arg in expr.args)...))
walkexpr(f, x) = f(x)

# Insert the keyword argument `maxiters` into calls to `solve` and `Pixie.solve`
# with default value `10^5` if it is not already present.
function insert_maxiters(expr)
    maxiters_default = 10^5

    expr = walkexpr(expr) do x
        if x isa Expr
            is_plain_solve = x.head === Symbol("call") && x.args[1] === Symbol("solve")
            is_pixie_solve = (x.head === Symbol("call") && x.args[1] isa Expr &&
                              x.args[1].head === Symbol(".") &&
                              x.args[1].args[1] === Symbol("Pixie") &&
                              x.args[1].args[2] isa QuoteNode &&
                              x.args[1].args[2].value === Symbol("solve"))

            if is_plain_solve || is_pixie_solve
            # Do nothing if `maxiters` is already set as keyword argument...
            for arg in x.args
                if arg isa Expr && arg.head === Symbol("kw") && arg.args[1] === Symbol("maxiters")
                    return x
                end
            end

            # ...and insert it otherwise.
            push!(x.args, Expr(Symbol("kw"), Symbol("maxiters"), maxiters_default))
            end
        end
        return x
    end

    return expr
end

# Replace assignments to `key` in `expr` by `key = val` for all `(key,val)` in `kwargs`.
function replace_assignments(expr; kwargs...)
    # replace explicit and keyword assignemnts
    expr = walkexpr(expr) do x
        if x isa Expr
            for (key,val) in kwargs
            if (x.head === Symbol("=") || x.head === :kw) && x.args[1] === Symbol(key)
                x.args[2] = :( $val )
                # dump(x)
            end
            end
        end
        return x
    end

    return expr
end

# find a (keyword or common) assignment to `destination` in `expr`
# and return the assigned value
function find_assignment(expr, destination)
    # declare result to be able to assign to it in the closure
    local result

    # find explicit and keyword assignemnts
    walkexpr(expr) do x
        if x isa Expr
            if (x.head === Symbol("=") || x.head === :kw) && x.args[1] === Symbol(destination)
                result = x.args[2]
                # dump(x)
            end
        end
        return x
    end

    result
end

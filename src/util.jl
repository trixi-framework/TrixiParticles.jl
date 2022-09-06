# Print informative message at startup
function print_startup_message()
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

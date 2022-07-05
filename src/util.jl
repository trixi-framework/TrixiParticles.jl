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

# Enable debug timings `@trixi_timeit timer() "name" stuff...`.
# This allows us to disable timings completely by executing
# `TimerOutputs.disable_debug_timings(Trixi)`
# and to enable them again by executing
# `TimerOutputs.enable_debug_timings(Trixi)`
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
# Copied from [Trixi.jl](https://github.com/trixi-framework/Trixi.jl).
macro pixie_timeit(timer_output, label, expr)
    timeit_block = quote
      if timeit_debug_enabled()
        local to = $(esc(timer_output))
        local enabled = to.enabled
        if enabled
          local accumulated_data = $(TimerOutputs.push!)(to, $(esc(label)))
        end
        local b₀ = $(TimerOutputs.gc_bytes)()
        local t₀ = $(TimerOutputs.time_ns)()
      end
      local val = $(esc(expr))
      if timeit_debug_enabled() && enabled
        $(TimerOutputs.do_accumulate!)(accumulated_data, t₀, b₀)
        $(TimerOutputs.pop!)(to)
      end
      val
    end
  end

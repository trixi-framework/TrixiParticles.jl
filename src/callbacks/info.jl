mutable struct InfoCallback
    start_time::Float64
    interval::Int
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:InfoCallback})
    @nospecialize cb # reduce precompilation time

    callback = cb.affect!
    print(io, "InfoCallback(interval=", callback.interval, ")")
end

function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{<:Any, <:InfoCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        callback = cb.affect!

        setup = [
            "interval" => callback.interval
        ]
        summary_box(io, "InfoCallback", setup)
    end
end

"""
    InfoCallback()

Create and return a callback that prints a human-readable summary of the simulation setup at the
beginning of a simulation and then resets the timer. When the returned callback is executed
directly, the current timer values are shown.
"""
function InfoCallback(; interval=0, reset_threads=true)
    info_callback = InfoCallback(0.0, interval)

    function initialize(cb, u, t, integrator)
        initialize_info_callback(cb, u, t, integrator;
                                 reset_threads)
    end

    DiscreteCallback(info_callback, info_callback,
                     save_positions=(false, false),
                     initialize=initialize)
end

# condition
function (info_callback::InfoCallback)(u, t, integrator)
    (; interval) = info_callback

    return interval != 0 &&
           integrator.stats.naccept % interval == 0 ||
           isfinished(integrator)
end

# affect!
function (info_callback::InfoCallback)(integrator)
    if isfinished(integrator)
        print_summary(integrator)
    else
        t = integrator.t
        t_initial = first(integrator.sol.prob.tspan)
        t_final = last(integrator.sol.prob.tspan)
        sim_time_percentage = (t - t_initial) / (t_final - t_initial) * 100
        runtime_absolute = 1.0e-9 * (time_ns() - info_callback.start_time)
        println(rpad(@sprintf("#timesteps: %6d │ Δt: %.4e │ sim. time: %.4e (%5.3f%%)",
                              integrator.stats.naccept, integrator.dt, t,
                              sim_time_percentage), 71) *
                @sprintf("│ run time: %.4e s", runtime_absolute))
    end

    # Tell OrdinaryDiffEq that u has not been modified
    u_modified!(integrator, false)

    return nothing
end

# Print information about the current simulation setup
# Note: This is called *after* all initialization is done, but *before* the first time step
function initialize_info_callback(discrete_callback, u, t, integrator;
                                  reset_threads=true)
    info_callback = discrete_callback.affect!

    # Optionally reset Polyester.jl threads. See
    # https://github.com/trixi-framework/Trixi.jl/issues/1583
    # https://github.com/JuliaSIMD/Polyester.jl/issues/30
    if reset_threads
        Polyester.reset_threads!()
    end

    print_startup_message()

    io = stdout
    io_context = IOContext(io,
                           :compact => false,
                           :key_width => 30,
                           :total_width => 100,
                           :indentation_level => 0)

    semi = integrator.p
    show(io_context, MIME"text/plain"(), semi)
    println(io, "\n")
    foreach_system(semi) do system
        show(io_context, MIME"text/plain"(), system)
        println(io, "\n")
    end

    callbacks = integrator.opts.callback
    # Assuming that callbacks is always a CallbackSet
    for cb in callbacks.continuous_callbacks
        show(io_context, MIME"text/plain"(), cb)
        println(io, "\n")
    end
    for cb in callbacks.discrete_callbacks
        # Do not show ourselves
        cb.affect! === info_callback && continue

        show(io_context, MIME"text/plain"(), cb)
        println(io, "\n")
    end

    # Time integration
    setup = Pair{String, Any}["Start time" => first(integrator.sol.prob.tspan),
                              "Final time" => last(integrator.sol.prob.tspan),
                              "time integrator" => integrator.alg |> typeof |> nameof,
                              "adaptive" => integrator.opts.adaptive]
    if integrator.opts.adaptive
        push!(setup,
              "abstol" => integrator.opts.abstol,
              "reltol" => integrator.opts.reltol,
              "controller" => integrator.opts.controller)
    end
    summary_box(io, "Time integration", setup)
    println()

    # Technical details
    setup = Pair{String, Any}["Julia version" => VERSION,
                              "parallelization backend" => semi.parallelization_backend |> typeof |> nameof,
                              "#threads" => Threads.nthreads()]
    summary_box(io, "Environment information", setup)
    println()
    println()

    reset_timer!(timer())

    # Save current time as start_time
    info_callback.start_time = time_ns()

    return nothing
end

# The following are functions to format summary output.
# This is all copied from Trixi.jl.
#
# Format a key/value pair for output from the InfoCallback
function format_key_value_line(key::AbstractString, value::AbstractString, key_width,
                               total_width;
                               indentation_level=0, guide='…', filler='…', prefix="│ ",
                               suffix=" │")
    @assert key_width < total_width
    line = prefix
    # Indent the key as requested (or not at all if `indentation_level == 0`)
    indentation = prefix^indentation_level
    reduced_key_width = key_width - length(indentation)
    squeezed_key = indentation * squeeze(key, reduced_key_width, filler=filler)
    line *= squeezed_key
    line *= ": "
    short = key_width - length(squeezed_key)
    if short <= 1
        line *= " "
    else
        line *= guide^(short - 1) * " "
    end
    value_width = total_width - length(prefix) - length(suffix) - key_width - 2
    squeezed_value = squeeze(value, value_width, filler=filler)
    line *= squeezed_value
    short = value_width - length(squeezed_value)
    line *= " "^short
    line *= suffix

    @assert length(line)==total_width "should not happen: algorithm error!"

    return line
end

function format_key_value_line(key, value, args...; kwargs...)
    format_key_value_line(string(key), string(value), args...; kwargs...)
end

# Squeeze a string to fit into a maximum width by deleting characters from the center
function squeeze(message, max_width; filler::Char='…')
    @assert max_width>=3 "squeezing works only for a minimum `max_width` of 3"

    length(message) <= max_width && return message

    keep_front = div(max_width, 2)
    keep_back = div(max_width, 2) - (isodd(max_width) ? 0 : 1)
    remove_back = length(message) - keep_front
    remove_front = length(message) - keep_back
    squeezed = (chop(message, head=0, tail=remove_back)
                * filler *
                chop(message, head=remove_front, tail=0))

    @assert length(squeezed)==max_width "`$(length(squeezed)) != $max_width` should not happen: algorithm error!"

    return squeezed
end

# Print a summary with a box around it with a given heading and a setup of key=>value pairs
function summary_box(io::IO, heading, setup=[])
    summary_header(io, heading)
    for (key, value) in setup
        summary_line(io, key, value)
    end
    summary_footer(io)
end

function summary_header(io, heading; total_width=100, indentation_level=0)
    total_width = get(io, :total_width, total_width)
    indentation_level = get(io, :indentation_level, indentation_level)

    @assert indentation_level>=0 "indentation level may not be negative"

    # If indentation level is greater than zero, we assume the header has already been printed
    indentation_level > 0 && return

    # Print header
    println(io, "┌" * "─"^(total_width - 2) * "┐")
    println(io, "│ " * heading * " "^(total_width - length(heading) - 4) * " │")
    println(io, "│ " * "═"^length(heading) * " "^(total_width - length(heading) - 4) * " │")
end

function summary_line(io, key, value; key_width=30, total_width=100, indentation_level=0)
    # Printing is not performance-critical, so we can use `@nospecialize` to reduce latency
    @nospecialize value # reduce precompilation time

    key_width = get(io, :key_width, key_width)
    total_width = get(io, :total_width, total_width)
    indentation_level = get(io, :indentation_level, indentation_level)

    s = format_key_value_line(key, value, key_width, total_width,
                              indentation_level=indentation_level)

    println(io, s)
end

function summary_footer(io; total_width=100, indentation_level=0)
    total_width = get(io, :total_width, 100)
    indentation_level = get(io, :indentation_level, 0)

    if indentation_level == 0
        s = "└" * "─"^(total_width - 2) * "┘"
    else
        s = ""
    end

    print(io, s)
end

function print_summary(integrator)
    println("─"^100)
    println("Trixi simulation finished.  Final time: ", integrator.t,
            "  Time steps: ", integrator.stats.naccept, " (accepted), ",
            integrator.iter, " (total)")
    println("─"^100)
    println()

    # Print timer
    TimerOutputs.complement!(timer())
    print_timer(timer(), title="TrixiParticles.jl",
                allocations=true, linechars=:unicode, compact=false)
    println()
end

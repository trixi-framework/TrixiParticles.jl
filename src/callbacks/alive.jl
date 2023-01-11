mutable struct AliveCallback
    start_time::Float64
    alive_interval::Int
end


function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:AliveCallback})
    @nospecialize cb # reduce precompilation time

    alive_callback = cb.affect!
    print(io, "AliveCallback(alive_interval=", alive_callback.alive_interval, ")")
end

function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{<:Any, <:AliveCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        alive_callback = cb.affect!

        setup = [
                "interval" => alive_callback.alive_interval,
                ]
        summary_box(io, "AliveCallback", setup)
    end
end


"""
    AliveCallback(; alive_interval=0)

Inexpensive callback showing that a simulation is still running by printing
some information such as the current time to the screen every `alive_interval` time steps.
"""
function AliveCallback(; alive_interval=0)
    alive_callback = AliveCallback(0.0, alive_interval)

    DiscreteCallback(alive_callback, alive_callback,
                     save_positions=(false, false),
                     initialize=initialize)
end


# condition
function (alive_callback::AliveCallback)(u, t, integrator)
    @unpack alive_interval = alive_callback

    return alive_interval == 0 ||
        integrator.destats.naccept % alive_interval == 0 ||
        isfinished(integrator)
end

# affect!
function (alive_callback::AliveCallback)(integrator)
    if isfinished(integrator)
        println("─"^100)
        println("Pixie simulation finished.  Final time: ", integrator.t,
                "  Time steps: ", integrator.destats.naccept, " (accepted), ", integrator.iter, " (total)")
        println("─"^100)
        println()
    else
        runtime_absolute = 1.0e-9 * (time_ns() - alive_callback.start_time)
        @printf("#timesteps: %6d │ Δt: %.4e │ sim. time: %.4e │ run time: %.4e s\n",
                integrator.destats.naccept, integrator.dt, integrator.t, runtime_absolute)
    end

    # Tell OrdinaryDiffEq that u has not been modified
    u_modified!(integrator, false)

    return nothing
end


function initialize(discrete_callback, u, t, integrator)
    # Save current time as start_time
    alive_callback = discrete_callback.affect!
    alive_callback.start_time = time_ns()

    return nothing
end


@inline function isfinished(integrator)
    # Checking for floating point equality is OK here as `DifferentialEquations.jl`
    # sets the time exactly to the final time in the last iteration
    return integrator.t == last(integrator.sol.prob.tspan) ||
           isempty(integrator.opts.tstops) ||
           integrator.iter == integrator.opts.maxiters
end
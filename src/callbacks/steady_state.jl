"""
    SteadyStateCallback(; interval::Integer=0, dt=0.0, interval_size::Integer=10,
                        abstol=1.0e-8, reltol=1.0e-6)

Terminates the integration when the residual of the change in kinetic energy
falls below the threshold specified by `abstol, reltol`.

# Keywords
- `interval=0`:     Check steady state condition every `interval` time steps.
- `dt=0.0`:         Check steady state condition  in regular intervals of `dt` in terms
                    of integration time by adding additional `tstops`.
- `interval_size`:  The interval in which the change of the kinetic energy is considered.
                    `interval_size` is a (integer) multiple of `interval` or `dt`.
- `abstol`:         Absolute tolerance.
- `reltol`:         Relative tolerance.
"""
mutable struct SteadyStateCallback{I, RealT <: Real}
    interval      :: I
    abstol        :: RealT
    reltol        :: RealT
    previous_ekin :: Vector{Float64}
    interval_size :: Int
end

function SteadyStateCallback(; interval::Integer=0, dt=0.0, interval_size::Integer=10,
                             abstol=1.0e-8, reltol=1.0e-6)
    abstol, reltol = promote(abstol, reltol)

    if dt > 0 && interval > 0
        throw(ArgumentError("setting both `interval` and `dt` is not supported"))
    end

    if dt > 0
        interval = Float64(dt)
    end

    steady_state_callback = SteadyStateCallback(interval, abstol, reltol, [Inf64],
                                                interval_size)

    if dt > 0
        return PeriodicCallback(steady_state_callback, dt, save_positions=(false, false),
                                final_affect=true)
    else
        return DiscreteCallback(steady_state_callback, steady_state_callback,
                                save_positions=(false, false))
    end
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:SteadyStateCallback})
    @nospecialize cb # reduce precompilation time

    cb_ = cb.affect!

    print(io, "SteadyStateCallback(abstol=", cb_.abstol, ", ", "reltol=", cb_.reltol, ")")
end

function Base.show(io::IO,
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:SteadyStateCallback}})
    @nospecialize cb # reduce precompilation time

    cb_ = cb.affect!.affect!

    print(io, "SteadyStateCallback(abstol=", cb_.abstol, ", reltol=", cb_.reltol, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:SteadyStateCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        cb_ = cb.affect!

        setup = ["absolute tolerance" => cb_.abstol,
            "relative tolerance" => cb_.reltol,
            "interval" => cb_.interval,
            "interval size" => cb_.interval_size]
        summary_box(io, "SteadyStateCallback", setup)
    end
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:SteadyStateCallback}})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        cb_ = cb.affect!.affect!

        setup = ["absolute tolerance" => cb_.abstol,
            "relative tolerance" => cb_.reltol,
            "interval" => cb_.interval,
            "interval_size" => cb_.interval_size]
        summary_box(io, "SteadyStateCallback", setup)
    end
end

# `affect!` (`PeriodicCallback`)
function (cb::SteadyStateCallback)(integrator)
    steady_state_condition(cb, integrator) || return nothing

    print_summary(integrator)

    terminate!(integrator)
end

# `affect!` (`DiscreteCallback`)
function (cb::SteadyStateCallback{Int})(integrator)
    print_summary(integrator)

    terminate!(integrator)
end

# `condition` (`DiscreteCallback`)
function (steady_state_callback::SteadyStateCallback)(vu_ode, t, integrator)
    return steady_state_condition(steady_state_callback, integrator)
end

@inline function steady_state_condition(cb, integrator)
    (; abstol, reltol, previous_ekin, interval_size) = cb

    vu_ode = integrator.u
    v_ode, u_ode = vu_ode.x
    semi = integrator.p

    # Calculate kinetic energy
    ekin = 0.0
    foreach_system(semi) do system
        v = wrap_v(v_ode, system, semi)

        # Calculate kintetic energy
        for particle in each_moving_particle(system)
            velocity = current_velocity(v, system, particle)
            ekin += 0.5 * system.mass[particle] * dot(velocity, velocity)
        end
    end

    push!(previous_ekin, ekin)

    terminate = false

    if interval_size_condition(cb, integrator)
        popfirst!(previous_ekin)

        # Calculate MSE only over the `interval_size`
        mse = 0.0
        for index in 1:interval_size
            mse += (previous_ekin[index] - ekin)^2
        end

        mse /= interval_size

        threshold = abstol + reltol * ekin

        terminate = mse <= threshold
    end

    return terminate
end

# `DiscreteCallback`
@inline function interval_size_condition(cb::SteadyStateCallback{Int}, integrator)
    return integrator.stats.naccept > 0 &&
           round(integrator.stats.naccept / cb.interval) > cb.interval_size
end

# `PeriodicCallback`
@inline function interval_size_condition(cb::SteadyStateCallback, integrator)
    return integrator.stats.naccept > 0 &&
           round(Int, integrator.t / cb.interval) > cb.interval_size
end

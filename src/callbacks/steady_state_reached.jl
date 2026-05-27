"""
    SteadyStateReachedCallback(; interval::Integer=0, dt=0.0,
                               interval_size::Integer=10, abstol=1.0e-8, reltol=1.0e-6)

Terminates the integration when the change of kinetic energy between time steps
falls below the threshold specified by `abstol + reltol * ekin`,
where `ekin` is the total kinetic energy of the simulation.

# Keywords
- `interval=0`:     Check steady state condition every `interval` time steps.
                    Use either `interval` or `dt`.
- `dt=0.0`:         Check steady state condition in regular intervals of `dt` in terms
                    of integration time by adding additional `tstops`
                    (note that this may change the solution).
                    Use either `interval` or `dt`.
- `interval_size`:  The number of callback evaluations over which the change of the
                    kinetic energy is considered.
- `abstol`:         Absolute tolerance.
- `reltol`:         Relative tolerance.
"""
struct SteadyStateReachedCallback{I, ELTYPE <: Real}
    interval      :: I
    abstol        :: ELTYPE
    reltol        :: ELTYPE
    previous_ekin :: Vector{ELTYPE}
    interval_size :: Int
end

function SteadyStateReachedCallback(; interval::Integer=0, dt=0.0,
                                    interval_size::Integer=10, abstol=1.0e-8, reltol=1.0e-6)
    if interval < 0
        throw(ArgumentError("`interval` must be non-negative"))
    end

    if dt > 0 && interval > 0
        throw(ArgumentError("setting both `interval` and `dt` is not supported"))
    end

    if dt <= 0 && interval == 0
        throw(ArgumentError("either `interval` or `dt` must be set to a positive value"))
    end

    if interval_size <= 0
        throw(ArgumentError("`interval_size` must be positive"))
    end

    abstol, reltol = float.(promote(abstol, reltol))
    ELTYPE = typeof(abstol)

    interval_ = if dt > 0
        convert(ELTYPE, dt)
    else
        Int(interval)
    end

    steady_state_callback = SteadyStateReachedCallback(interval_, abstol, reltol,
                                                       [convert(ELTYPE, Inf)],
                                                       interval_size)

    if dt > 0
        return PeriodicCallback(steady_state_callback, dt,
                                initialize=(initialize_steady_state_callback!),
                                save_positions=(false, false))
    else
        return DiscreteCallback(steady_state_callback, steady_state_callback,
                                save_positions=(false, false),
                                initialize=(initialize_steady_state_callback!))
    end
end

function initialize_steady_state_callback!(cb, u, t, integrator)
    # The `SteadyStateReachedCallback` is either `cb.affect!` (with `DiscreteCallback`)
    # or `cb.affect!.affect!` (with `PeriodicCallback`).
    # Let recursive dispatch handle this.
    initialize_steady_state_callback!(cb.affect!, u, t, integrator)
end

function initialize_steady_state_callback!(cb::SteadyStateReachedCallback, u, t, integrator)
    empty!(cb.previous_ekin)
    push!(cb.previous_ekin, convert(eltype(cb.previous_ekin), Inf))

    return cb
end

# `affect!` (`PeriodicCallback`)
function (cb::SteadyStateReachedCallback)(integrator)
    if !steady_state_condition!(cb, integrator)
        u_modified!(integrator, false)
        return cb
    end

    print_summary(integrator)

    terminate!(integrator)

    return cb
end

# `affect!` (`DiscreteCallback`)
function (cb::SteadyStateReachedCallback{Int})(integrator)
    print_summary(integrator)

    terminate!(integrator)

    return cb
end

# `condition` (`DiscreteCallback`)
function (cb::SteadyStateReachedCallback{Int})(vu_ode, t, integrator)
    if !condition_integrator_interval(integrator, cb.interval; save_final_solution=false)
        return false
    end

    return steady_state_condition!(cb, integrator)
end

function (steady_state_callback::SteadyStateReachedCallback)(vu_ode, t, integrator)
    return steady_state_condition!(steady_state_callback, integrator)
end

@inline function steady_state_condition!(cb, integrator)
    (; abstol, reltol, previous_ekin, interval_size) = cb

    @trixi_timeit timer() "update dvdu" begin
        # Don't create sub-timers here to avoid cluttering the timer output
        @notimeit timer() dvdu_ode = get_dvdu(integrator)
        dv_ode, du_ode = dvdu_ode.x
    end

    v_ode, u_ode = integrator.u.x
    semi = integrator.p.semi

    # Calculate kinetic energy
    ekin = sum(semi.systems) do system
        return kinetic_energy(system, dv_ode, du_ode, v_ode, u_ode, semi, 0)
    end

    if length(previous_ekin) == interval_size
        # Calculate MSE only over the `interval_size`
        mse = sum(1:interval_size) do index
            return (previous_ekin[index] - ekin)^2 / interval_size
        end

        if mse <= abstol + reltol * ekin
            return true
        end

        # Pop old kinetic energy
        popfirst!(previous_ekin)
    end

    # Add current kinetic energy
    push!(previous_ekin, ekin)

    return false
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:SteadyStateReachedCallback})
    @nospecialize cb # reduce precompilation time

    cb_ = cb.affect!

    print(io, "SteadyStateReachedCallback(interval=", cb_.interval,
          ", abstol=", cb_.abstol, ", reltol=", cb_.reltol, ")")
end

function Base.show(io::IO,
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:SteadyStateReachedCallback}})
    @nospecialize cb # reduce precompilation time

    cb_ = cb.affect!.affect!

    print(io, "SteadyStateReachedCallback(dt=", cb_.interval,
          ", abstol=", cb_.abstol, ", reltol=", cb_.reltol, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:SteadyStateReachedCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        cb_ = cb.affect!

        setup = ["interval" => cb_.interval,
            "interval size" => cb_.interval_size,
            "absolute tolerance" => cb_.abstol,
            "relative tolerance" => cb_.reltol]
        summary_box(io, "SteadyStateReachedCallback", setup)
    end
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:SteadyStateReachedCallback}})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        cb_ = cb.affect!.affect!

        setup = ["dt" => cb_.interval,
            "interval size" => cb_.interval_size,
            "absolute tolerance" => cb_.abstol,
            "relative tolerance" => cb_.reltol]
        summary_box(io, "SteadyStateReachedCallback", setup)
    end
end

"""
    SteadyStateReachedCallback(; interval::Integer=0, dt=0.0,
                               interval_size::Integer=10, abstol=1.0e-8, reltol=1.0e-6)

Terminates the integration when the change of kinetic energy between time steps
falls below the threshold specified by `abstol + reltol * ekin`,
where `ekin` is the total kinetic energy of the simulation.

# Keywords
- `interval=0`:     Check steady state condition every `interval` time steps.
- `dt=0.0`:         Check steady state condition in regular intervals of `dt` in terms
                    of integration time by adding additional `tstops`
                    (note that this may change the solution).
- `interval_size`:  The interval in which the change of the kinetic energy is considered.
                    `interval_size` is a (integer) multiple of `interval` or `dt`.
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
    ELTYPE = eltype(abstol)
    abstol, reltol = promote(abstol, reltol)

    if dt > 0 && interval > 0
        throw(ArgumentError("setting both `interval` and `dt` is not supported"))
    end

    if dt > 0
        interval = convert(ELTYPE, dt)
    end

    steady_state_callback = SteadyStateReachedCallback(interval, abstol, reltol,
                                                       [convert(ELTYPE, Inf)],
                                                       interval_size)

    if dt > 0
        return PeriodicCallback(steady_state_callback, dt, save_positions=(false, false),
                                final_affect=true)
    else
        return DiscreteCallback(steady_state_callback, steady_state_callback,
                                save_positions=(false, false))
    end
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:SteadyStateReachedCallback})
    @nospecialize cb # reduce precompilation time

    cb_ = cb.affect!

    print(io, "SteadyStateReachedCallback(abstol=", cb_.abstol, ", ", "reltol=", cb_.reltol,
          ")")
end

function Base.show(io::IO,
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:SteadyStateReachedCallback}})
    @nospecialize cb # reduce precompilation time

    cb_ = cb.affect!.affect!

    print(io, "SteadyStateReachedCallback(abstol=", cb_.abstol, ", reltol=", cb_.reltol,
          ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:SteadyStateReachedCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        cb_ = cb.affect!

        setup = ["absolute tolerance" => cb_.abstol,
            "relative tolerance" => cb_.reltol,
            "interval" => cb_.interval,
            "interval size" => cb_.interval_size]
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

        setup = ["absolute tolerance" => cb_.abstol,
            "relative tolerance" => cb_.reltol,
            "interval" => cb_.interval,
            "interval_size" => cb_.interval_size]
        summary_box(io, "SteadyStateReachedCallback", setup)
    end
end

# `affect!` (`PeriodicCallback`)
function (cb::SteadyStateReachedCallback)(integrator)
    steady_state_condition!(cb, integrator) || return nothing

    print_summary(integrator)

    terminate!(integrator)
end

# `affect!` (`DiscreteCallback`)
function (cb::SteadyStateReachedCallback{Int})(integrator)
    print_summary(integrator)

    terminate!(integrator)
end

# `condition` (`DiscreteCallback`)
function (steady_state_callback::SteadyStateReachedCallback)(vu_ode, t, integrator)
    return steady_state_condition!(steady_state_callback, integrator)
end

@inline function steady_state_condition!(cb, integrator)
    (; abstol, reltol, previous_ekin, interval_size) = cb

    vu_ode = integrator.u
    v_ode, u_ode = vu_ode.x
    dv_ode, du_ode = get_du(integrator).x
    semi = integrator.p

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

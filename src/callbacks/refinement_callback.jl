mutable struct ParticleRefinementCallback{I}
    interval           :: I
    n_candidates       :: Int
    n_childs           :: Int
    ranges_u_cache     :: NTuple
    ranges_v_cache     :: NTuple
    eachparticle_cache :: NTuple

    # internal `resize!`able storage
    _u_ode::Vector{Float64}
    _v_ode::Vector{Float64}
end

function ParticleRefinementCallback(; interval::Integer=-1, dt=0.0)
    if dt > 0 && interval !== -1
        throw(ArgumentError("Setting both interval and dt is not supported!"))
    end

    # Update in intervals in terms of simulation time
    if dt > 0
        interval = Float64(dt)

        # Update every time step (default)
    elseif interval == -1
        interval = 1
    end

    refinement_callback = ParticleRefinementCallback(interval, 0, 0, (), (), (), [0.0], [0.0])

    if dt > 0
        # Add a `tstop` every `dt`, and save the final solution.
        return PeriodicCallback(refinement_callback, dt,
                                initialize=initial_refinement!,
                                save_positions=(false, false))
    else
        # The first one is the condition, the second the affect!
        return DiscreteCallback(refinement_callback, refinement_callback,
                                initialize=initial_refinement!,
                                save_positions=(false, false))
    end
end

# initialize
function initial_refinement!(cb, u, t, integrator)
    # The `ParticleRefinementCallback` is either `cb.affect!` (with `DiscreteCallback`)
    # or `cb.affect!.affect!` (with `PeriodicCallback`).
    # Let recursive dispatch handle this.

    initial_refinement!(cb.affect!, u, t, integrator)
end

function initial_refinement!(cb::ParticleRefinementCallback, u, t, integrator)
    cb(integrator)
end

# condition
function (refinement_callback::ParticleRefinementCallback)(u, t, integrator)
    (; interval) = refinement_callback

    # With error-based step size control, some steps can be rejected. Thus,
    #   `integrator.iter >= integrator.stats.naccept`
    #    (total #steps)       (#accepted steps)
    # We need to check the number of accepted steps since callbacks are not
    # activated after a rejected step.
    return integrator.stats.naccept % interval == 0
end

# affect
function (refinement_callback::ParticleRefinementCallback)(integrator)
    t = integrator.t
    semi = integrator.p
    v_ode, u_ode = integrator.u.x

    refinement!(v_ode, u_ode, semi, refinement_callback)

    update_systems_and_nhs(v_ode, u_ode, semi, t)

    # Tell OrdinaryDiffEq that u has been modified
    u_modified!(integrator, true)

    return integrator
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:ParticleRefinementCallback})
    @nospecialize cb # reduce precompilation time
    print(io, "ParticleRefinementCallback(interval=", (cb.affect!.interval), ")")
end

function Base.show(io::IO,
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:ParticleRefinementCallback}})
    @nospecialize cb # reduce precompilation time
    print(io, "ParticleRefinementCallback(dt=", cb.affect!.affect!.interval, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:ParticleRefinementCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        refinement_cb = cb.affect!
        setup = [
            "interval" => refinement_cb.interval,
        ]
        summary_box(io, "ParticleRefinementCallback", setup)
    end
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:ParticleRefinementCallback}})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        refinement_cb = cb.affect!.affect!
        setup = [
            "dt" => refinement_cb.interval,
        ]
        summary_box(io, "ParticleRefinementCallback", setup)
    end
end

mutable struct ParticleRefinementCallback{I}
    interval           :: I
    n_candidates       :: Vector{Int}
    n_childs           :: Vector{Int}
    ranges_u_cache     :: RU
    ranges_v_cache     :: RV
    eachparticle_cache :: RP

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

    update_callback! = ParticleRefinementCallback(interval)

    if dt > 0 && update
        # Add a `tstop` every `dt`, and save the final solution.
        return PeriodicCallback(update_callback!, dt,
                                initialize=initial_update!,
                                save_positions=(false, false))
    else
        # The first one is the condition, the second the affect!
        return DiscreteCallback(update_callback!, update_callback!,
                                initialize=initial_update!,
                                save_positions=(false, false))
    end
end

# initialize
function initial_update!(cb, u, t, integrator)
    # The `ParticleRefinementCallback` is either `cb.affect!` (with `DiscreteCallback`)
    # or `cb.affect!.affect!` (with `PeriodicCallback`).
    # Let recursive dispatch handle this.

    initial_update!(cb.affect!, u, t, integrator)
end

function initial_update!(cb::ParticleRefinementCallback, u, t, integrator)
    cb.update && cb(integrator)
end

# condition
function (update_callback!::ParticleRefinementCallback)(u, t, integrator)
    (; interval, update) = update_callback!

    # With error-based step size control, some steps can be rejected. Thus,
    #   `integrator.iter >= integrator.stats.naccept`
    #    (total #steps)       (#accepted steps)
    # We need to check the number of accepted steps since callbacks are not
    # activated after a rejected step.
    return update && (integrator.stats.naccept % interval == 0)
end

# affect
function (update_callback!::ParticleRefinementCallback)(integrator)
    t = integrator.t
    semi = integrator.p
    v_ode, u_ode = integrator.u.x

    # Update quantities that are stored in the systems. These quantities (e.g. pressure)
    # still have the values from the last stage of the previous step if not updated here.
    refinement!(v_ode, u_ode, semi, callback)

    update_systems_and_nhs(v_ode, u_ode, semi, t)

    # Tell OrdinaryDiffEq that u has been modified
    u_modified!(integrator, true)

    return integrator
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:ParticleRefinementCallback})
    @nospecialize cb # reduce precompilation time
    print(io, "ParticleRefinementCallback(interval=",
          (cb.affect!.update ? cb.affect!.interval : "-"),
          ")")
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
        update_cb = cb.affect!
        setup = [
            "interval" => update_cb.update ? update_cb.interval : "-",
            "update" => update_cb.update ? "yes" : "no"
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
        update_cb = cb.affect!.affect!
        setup = [
            "dt" => update_cb.interval,
            "update" => update_cb.update ? "yes" : "no"
        ]
        summary_box(io, "ParticleRefinementCallback", setup)
    end
end

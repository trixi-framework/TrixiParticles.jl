struct UpdateCallback{I}
    interval::I
end

"""
    UpdateCallback(; interval::Integer, dt=0.0)

Callback to update quantities either at the end of every `interval` time steps or
in intervals of `dt` in terms of integration time by adding additional `tstops`
(note that this may change the solution).

Rigid contact with tangential spring history requires exactly one
`UpdateCallback(interval=1)`. Sparse step intervals and `dt`-based schedules cannot advance
that path-dependent state correctly and are rejected when such contact is present.

# Keywords
- `interval=1`: Update quantities at the end of every `interval` time steps.
- `dt`: Update quantities in regular intervals of `dt` in terms of integration time
        by adding additional `tstops` (note that this may change the solution).
"""
function UpdateCallback(; interval::Integer=-1, dt=0.0)
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

    update_callback! = UpdateCallback(interval)

    if dt > 0
        # Add a `tstop` every `dt`
        return PeriodicCallback(update_callback!, dt,
                                initialize=(initial_update!),
                                save_positions=(false, false))
    else
        # The first one is the `condition`, the second the `affect!`
        return DiscreteCallback(update_callback!, update_callback!,
                                initialize=(initial_update!),
                                save_positions=(false, false))
    end
end

# `initialize`
function initial_update!(cb, u, t, integrator)
    # The `UpdateCallback` is either `cb.affect!` (with `DiscreteCallback`)
    # or `cb.affect!.affect!` (with `PeriodicCallback`).
    # Let recursive dispatch handle this.

    initial_update!(cb.affect!, u, t, integrator)
end

function initial_update!(cb::UpdateCallback, vu_ode, t, integrator)
    v_ode, u_ode = vu_ode.x
    semi = integrator.p.semi

    validate_rigid_contact_update_callbacks!(semi, integrator)

    # Tell the semidiscretization that the `UpdateCallback` is used
    semi.update_callback_used[] = true

    # If TLSPH is not integrated, the averaged velocity will be initialized in the
    # split integration.
    if semi.integrate_tlsph[]
        foreach_system(semi) do system
            initialize_averaged_velocity!(system, v_ode, semi, t)
        end
    end

    return run_update_callback!(cb, integrator; initial=true)
end

# `condition`
function (update_callback!::UpdateCallback)(u, t, integrator)
    (; interval) = update_callback!

    return condition_integrator_interval(integrator, interval)
end

# `affect!`
function (callback::UpdateCallback)(integrator)
    return run_update_callback!(callback, integrator; initial=false)
end

function run_update_callback!(callback::UpdateCallback, integrator; initial)
    t = integrator.t
    semi = integrator.p.semi
    v_ode, u_ode = integrator.u.x

    # Contact history is endpoint state, not ODE stage state. Initialization discovers
    # contacts with zero elapsed time; subsequent calls use the last accepted step length.
    # In particular, `integrator.dt` can already contain the proposal for the next step.
    history_dt = initial ? zero(t) : t - integrator.tprev

    # An empty update without calling any of the functions below does not modify
    # the results of the right-hand side.
    # The functions that add a discontinuity call `derivative_discontinuity!` themselves.
    derivative_discontinuity!(integrator, false)

    @trixi_timeit timer() "update callback" begin
        # Update quantities that are stored in the systems. These quantities (e.g. pressure)
        # still have the values from the last stage of the previous step if not updated here.
        @trixi_timeit timer() "update systems and nhs" begin
            # Don't create sub-timers here to avoid cluttering the timer output
            @notimeit timer() update_systems_and_nhs(v_ode, u_ode, semi, t)
        end

        # Update open boundaries first, since particles might be activated or deactivated
        foreach_system(semi) do system
            update_open_boundary_eachstep!(system, v_ode, u_ode, semi, t, integrator)
        end

        foreach_system(semi) do system
            update_particle_packing(system, v_ode, u_ode, semi, integrator)
        end

        contact_history_changed = false
        foreach_system(semi) do system
            contact_history_changed |= update_rigid_contact_eachstep!(system, v_ode, u_ode,
                                                                      semi, t, history_dt)
        end

        # FSAL methods cache the endpoint derivative for reuse as the next first stage.
        # Tangential-history changes alter contact forces without changing `u`, so that
        # derivative must be recomputed.
        contact_history_changed && derivative_discontinuity!(integrator, true)

        # This is only used by the particle packing system and should be removed in the future
        foreach_system(semi) do system
            update_transport_velocity!(system, v_ode, semi, integrator)
        end

        foreach_system(semi) do system
            particle_shifting_from_callback!(u_ode, shifting_technique(system), system,
                                             v_ode, semi, integrator)
        end

        # If TLSPH is not integrated, the averaged velocity will be updated in the
        # split integration.
        if semi.integrate_tlsph[]
            foreach_system(semi) do system
                compute_averaged_velocity!(system, v_ode, semi, t)
            end
        end
    end

    return integrator
end

function validate_rigid_contact_update_callbacks!(semi, integrator)
    hasproperty(semi, :systems) || return semi
    any(system -> system isa RigidBodySystem &&
                  requires_update_callback(system, semi), semi.systems) || return semi

    UpdateCB = Union{DiscreteCallback{<:Any, <:UpdateCallback},
                     DiscreteCallback{<:Any, <:PeriodicCallbackAffect{<:UpdateCallback}}}
    # SciML wraps step-based and time-periodic callbacks differently. Normalize both forms
    # here so contact history has one unambiguous owner and one accepted-step schedule.
    callbacks = filter(cb -> cb isa UpdateCB,
                       integrator.opts.callback.discrete_callbacks)

    length(callbacks) == 1 ||
        throw(ArgumentError("rigid contact history requires exactly one `UpdateCallback`"))

    callback = only(callbacks)
    update_callback = callback.affect! isa UpdateCallback ? callback.affect! :
                      callback.affect!.affect!
    valid_schedule = update_callback.interval isa Integer && update_callback.interval == 1
    valid_schedule ||
        throw(ArgumentError("rigid contact history requires `UpdateCallback(interval=1)`"))

    if semi.parallelization_backend isa KernelAbstractions.GPU
        throw(ArgumentError("rigid contact history is not supported on GPU backends"))
    end

    return semi
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:UpdateCallback})
    @nospecialize cb # reduce precompilation time
    print(io, "UpdateCallback(interval=", cb.affect!.interval, ")")
end

function Base.show(io::IO,
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:UpdateCallback}})
    @nospecialize cb # reduce precompilation time
    print(io, "UpdateCallback(dt=", cb.affect!.affect!.interval, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:UpdateCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        update_cb = cb.affect!
        setup = [
            "interval" => update_cb.interval
        ]
        summary_box(io, "UpdateCallback", setup)
    end
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:UpdateCallback}})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        update_cb = cb.affect!.affect!
        setup = [
            "dt" => update_cb.interval
        ]
        summary_box(io, "UpdateCallback", setup)
    end
end

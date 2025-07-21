struct UpdateCallback{I}
    interval::I
end

"""
    UpdateCallback(; interval::Integer, dt=0.0)

Callback to update quantities either at the end of every `interval` time steps or
in intervals of `dt` in terms of integration time by adding additional `tstops`
(note that this may change the solution).

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
        # Add a `tstop` every `dt`, and save the final solution.
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

function initial_update!(cb::UpdateCallback, u, t, integrator)
    semi = integrator.p

    # Tell systems that `UpdateCallback` is used
    foreach_system(semi) do system
        update_callback_used!(system)
    end

    return cb(integrator)
end

# `condition`
function (update_callback!::UpdateCallback)(u, t, integrator)
    (; interval) = update_callback!

    return condition_integrator_interval(integrator, interval)
end

# `affect!`
function (update_callback!::UpdateCallback)(integrator)
    t = integrator.t
    semi = integrator.p
    v_ode, u_ode = integrator.u.x

    # Update quantities that are stored in the systems. These quantities (e.g. pressure)
    # still have the values from the last stage of the previous step if not updated here.
    update_systems_and_nhs(v_ode, u_ode, semi, t; update_from_callback=true)

    # Update open boundaries first, since particles might be activated or deactivated
    @trixi_timeit timer() "update open boundary" foreach_system(semi) do system
        update_open_boundary_eachstep!(system, v_ode, u_ode, semi, t)
    end

    @trixi_timeit timer() "update particle packing" foreach_system(semi) do system
        update_particle_packing(system, v_ode, u_ode, semi, integrator)
    end

    # Tell OrdinaryDiffEq that `u` has been modified
    u_modified!(integrator, true)

    return integrator
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

update_callback_used!(system) = system

struct ParticleRefinementCallback{I}
    interval::I
end

"""
    ParticleRefinementCallback(; interval::Integer, dt=0.0)

Callback to apply particle refinement to all fluid systems with a `ParticleRefinement`
either at the end of every `interval` time steps or in intervals of `dt` in terms of
integration time by adding additional `tstops` (note that this may change the solution).

Before the time integration starts, the callback builds the graded particle distribution
by applying `n_startup_rounds` refinement rounds (see `ParticleRefinement`).
Refinement can add particles to a system, in which case the integrator is resized.
After refinement, all quantities that are stored in the systems are recomputed,
so the order of this callback relative to other callbacks does not matter.

# Keywords
- `interval=1`: Apply refinement at the end of every `interval` time steps.
- `dt`: Apply refinement in regular intervals of `dt` in terms of integration time
        by adding additional `tstops` (note that this may change the solution).
"""
function ParticleRefinementCallback(; interval::Integer=-1, dt=0.0)
    if dt > 0 && interval !== -1
        throw(ArgumentError("setting both `interval` and `dt` is not supported"))
    end

    # Apply refinement in intervals in terms of simulation time
    if dt > 0
        interval = Float64(dt)

        # Apply refinement every time step (default)
    elseif interval == -1
        interval = 1
    end

    refinement_callback = ParticleRefinementCallback(interval)

    if dt > 0
        # Add a `tstop` every `dt`
        return PeriodicCallback(refinement_callback, dt,
                                initialize=(initial_refinement!),
                                save_positions=(false, false))
    else
        # The first one is the `condition`, the second the `affect!`
        return DiscreteCallback(refinement_callback, refinement_callback,
                                initialize=(initial_refinement!),
                                save_positions=(false, false))
    end
end

# `initialize`
function initial_refinement!(cb, u, t, integrator)
    # The `ParticleRefinementCallback` is either `cb.affect!` (with `DiscreteCallback`)
    # or `cb.affect!.affect!` (with `PeriodicCallback`).
    # Let recursive dispatch handle this.
    initial_refinement!(cb.affect!, u, t, integrator)
end

function initial_refinement!(cb::ParticleRefinementCallback, u, t, integrator)
    check_refinement_buffers(integrator.p.semi)

    @trixi_timeit timer() "refinement startup" begin
        apply_refinement!(refinement_startup!, integrator)
    end

    return integrator
end

# Splitting puts new particles into inactive slots of the `SystemBuffer` and merging
# deactivates particles. Without a buffer, refinement would only fail at the first split
# or merge, so check this before the startup.
# This is not checked in the system constructors, so that systems with refinement
# can be used without a buffer when they are never refined (e.g., in unit tests).
function check_refinement_buffers(semi)
    foreach_system(semi) do system
        isnothing(get_refinement(system)) && return

        if !(system.buffer isa SystemBuffer)
            throw(ArgumentError("particle refinement requires a `SystemBuffer`. " *
                                "Pass `buffer_size` to the " *
                                "`$(nameof(typeof(system)))` with `particle_refinement`."))
        end
    end

    return semi
end

# `condition`
function (refinement_callback::ParticleRefinementCallback)(u, t, integrator)
    (; interval) = refinement_callback

    return condition_integrator_interval(integrator, interval)
end

# `affect!`
function (refinement_callback::ParticleRefinementCallback)(integrator)
    @trixi_timeit timer() "refinement" apply_refinement!(refinement!, integrator)

    return integrator
end

# Apply `refinement_function!(semi, v_ode, u_ode)` and update the integrator
function apply_refinement!(refinement_function!, integrator)
    semi = integrator.p.semi
    v_ode, u_ode = integrator.u.x

    length_v = length(v_ode)
    length_u = length(u_ode)

    # Refinement uses the density and pressure to merge particles and to correct
    # the particle properties after shifting (Eq. 37). Before the first time step,
    # the summation density is not computed yet. Afterwards, the quantities are from
    # the last stage of the time step and not from the current positions.
    update_systems_and_nhs(v_ode, u_ode, semi, integrator.t)

    refinement_function!(semi, v_ode, u_ode)

    # Refinement resizes `v_ode` and `u_ode` when new particles don't fit
    # into the existing buffer (see `resize_systems!`)
    if length(v_ode) != length_v || length(u_ode) != length_u
        resize!(integrator, (length(v_ode), length(u_ode)))
    end

    # Recompute all quantities that are stored in the systems (e.g. the pressure),
    # since particles were added, removed and moved.
    # Otherwise, they would be stale until the next update, for example when
    # the `UpdateCallback` comes before this callback.
    update_systems_and_nhs(v_ode, u_ode, semi, integrator.t)

    # Shifting velocities that are only computed in the `UpdateCallback` are not
    # updated above (e.g. `ParticleShiftingTechnique` with `update_everystage=false`)
    foreach_system(semi) do system
        update_shifting_from_callback!(system, shifting_technique(system), v_ode, u_ode,
                                       semi)
    end

    # Tell OrdinaryDiffEq that the solution was modified
    derivative_discontinuity!(integrator, true)

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
            "interval" => refinement_cb.interval
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
            "dt" => refinement_cb.interval
        ]
        summary_box(io, "ParticleRefinementCallback", setup)
    end
end

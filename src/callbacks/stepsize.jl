struct StepsizeCallback{ISCONSTANT, ELTYPE}
    cfl_number::ELTYPE
end

@inline is_constant(::StepsizeCallback{ISCONSTANT}) where {ISCONSTANT} = ISCONSTANT

@doc raw"""
    StepsizeCallback(; cfl::Real)

Set the time step size according to a CFL condition if the time integration method isn't
adaptive itself.

The current implementation is using the simplest form of CFL condition, which chooses a
time step size that is constant during the simulation.
The step size is therefore only applied once at the beginning of the simulation.

The step size ``\Delta t`` is chosen as the minimum
```math
    \Delta t = \min(\Delta t_\eta, \Delta t_a, \Delta t_c),
```
where
```math
    \Delta t_\eta = 0.125 \, h^2 / \eta, \quad \Delta t_a = 0.25 \sqrt{h / \lVert g \rVert},
    \quad \Delta t_c = \text{CFL} \, h / c,
```
with ``\nu = \alpha h c / (2n + 4)``, where ``\alpha`` is the parameter of the viscosity
and ``n`` is the number of dimensions.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.

## References
[Antuono2012](@cite), [Adami2012](@cite), [Sun2017](@cite), [Antuono2015](@cite)
"""
function StepsizeCallback(; cfl::Real)
    # TODO adapt for non-constant CFL conditions
    is_constant = true
    stepsize_callback = StepsizeCallback{is_constant, typeof(cfl)}(cfl)

    # The first one is the `condition`, the second the `affect!`
    return DiscreteCallback(stepsize_callback, stepsize_callback,
                            save_positions=(false, false),
                            initialize=initialize_stepsize_callback)
end

function initialize_stepsize_callback(discrete_callback, u, t, integrator)
    stepsize_callback = discrete_callback.affect!

    semi = integrator.p.semi
    set_callbacks_used!(semi, integrator)

    stepsize_callback(integrator)
end

# `condition`
function (stepsize_callback::StepsizeCallback)(u, t, integrator)
    # Only apply the callback when the stepsize is not constant and the time integrator
    # is not adaptive.
    return !is_constant(stepsize_callback) && !integrator.opts.adaptive
end

# `affect!`
function (stepsize_callback::StepsizeCallback)(integrator)
    (; cfl_number) = stepsize_callback

    v_ode, u_ode = integrator.u.x
    semi = integrator.p.semi

    dt = @trixi_timeit timer() "calculate dt" calculate_dt(v_ode, u_ode, cfl_number, semi)

    set_proposed_dt!(integrator, dt)
    integrator.opts.dtmax = dt
    integrator.dtcache = dt
    uses_iisph_projection_dt(semi) && set_iisph_projection_dt!(semi, dt)

    # Tell OrdinaryDiffEq that `u` has not been modified
    u_modified!(integrator, false)

    return stepsize_callback
end

struct IISPHTimeStepCallback{R, W, P}
    require_fixed_step::R
    warm_start_pressure::W
    project_at_step_end::P
end

@doc raw"""
    IISPHTimeStepCallback(; require_fixed_step=true, warm_start_pressure=true,
                          project_at_step_end=false)

Synchronize the IISPH pressure projection step size with the current time step of
the OrdinaryDiffEq.jl integrator.

Use this callback when an [`ImplicitIncompressibleSPHSystem`](@ref) is integrated with
fixed-step OrdinaryDiffEq.jl schemes where the step size is passed to `solve`, e.g.
`solve(ode, CarpenterKennedy2N54(williamson_condition=false); dt, adaptive=false)`.

Adaptive time integration is currently rejected by default because rejected steps require
restoring IISPH pressure caches. This can be disabled with `require_fixed_step=false` for
experiments.

When `warm_start_pressure=true`, IISPH pressure initialization is damped once per accepted
time step and RK stages reuse the previous stage pressure as initial guess.

When `project_at_step_end=true`, RK stages evaluate the non-pressure right-hand side and
the callback runs the IISPH pressure projection once at the end of every accepted step.
"""
function IISPHTimeStepCallback(; require_fixed_step=true, warm_start_pressure=true,
                               project_at_step_end=false)
    callback = IISPHTimeStepCallback{typeof(require_fixed_step),
                                     typeof(warm_start_pressure),
                                     typeof(project_at_step_end)}(require_fixed_step,
                                                                  warm_start_pressure,
                                                                  project_at_step_end)

    return DiscreteCallback(callback, callback;
                            initialize=initialize_iisph_time_step_callback,
                            save_positions=(false, false))
end

function initialize_iisph_time_step_callback(discrete_callback, u, t, integrator)
    callback = discrete_callback.affect!
    initialize_iisph_pressure_warm_start!(integrator.p.semi,
                                          callback.warm_start_pressure)
    initialize_iisph_step_end_projection!(integrator.p.semi,
                                          callback.project_at_step_end)
    sync_iisph_projection_dt!(integrator;
                              require_fixed_step=callback.require_fixed_step)

    return discrete_callback
end

# `condition`
function (callback::IISPHTimeStepCallback)(u, t, integrator)
    return uses_iisph_projection_dt(integrator.p.semi)
end

# `affect!`
function (callback::IISPHTimeStepCallback)(integrator)
    sync_iisph_projection_dt!(integrator;
                              require_fixed_step=callback.require_fixed_step)

    if callback.project_at_step_end
        project_iisph_pressure_at_step_end!(integrator)
    end

    reset_iisph_pressure_warm_start!(integrator.p.semi,
                                     callback.warm_start_pressure)

    # The step-end projection mutates the accepted state.
    u_modified!(integrator, callback.project_at_step_end)

    return callback
end

struct IISPHTimeStepLimiter{R, W}
    require_fixed_step::R
    warm_start_pressure::W
end

@doc raw"""
    IISPHTimeStepLimiter(; require_fixed_step=true, warm_start_pressure=true)

Limiter object for OrdinaryDiffEq.jl Runge-Kutta algorithms exposing `stage_limiter!`
and `step_limiter!` keyword arguments.

The limiter does not alter the RK stage state. Use it together with
[`IISPHTimeStepCallback`](@ref) to synchronize the IISPH projection step size from
`integrator.dt` at RK stage boundaries when OrdinaryDiffEq.jl passes the integrator to the
limiter.

When `warm_start_pressure=true`, IISPH pressure initialization is damped once per accepted
time step and RK stages reuse the previous stage pressure as initial guess.
"""
function IISPHTimeStepLimiter(; require_fixed_step=true, warm_start_pressure=true)
    return IISPHTimeStepLimiter{typeof(require_fixed_step),
                                typeof(warm_start_pressure)}(require_fixed_step,
                                                             warm_start_pressure)
end

function (limiter::IISPHTimeStepLimiter)(u, integrator, p, t)
    # Some OrdinaryDiffEq.jl algorithms call stage limiters with the ODEFunction instead
    # of the integrator at selected stages. In that case there is no step size to sync.
    hasproperty(integrator, :dt) || return u
    hasproperty(p, :semi) || return u

    update_iisph_pressure_warm_start!(p.semi, limiter.warm_start_pressure,
                                      integrator, t)
    sync_iisph_projection_dt!(integrator, p;
                              require_fixed_step=limiter.require_fixed_step)

    return u
end

function sync_iisph_projection_dt!(integrator; require_fixed_step=true)
    return sync_iisph_projection_dt!(integrator, integrator.p; require_fixed_step)
end

function sync_iisph_projection_dt!(integrator, p; require_fixed_step=true)
    semi = p.semi
    uses_iisph_projection_dt(semi) || return integrator

    if require_fixed_step && integrator.opts.adaptive
        throw(ArgumentError("IISPH currently supports fixed-step OrdinaryDiffEq.jl " *
                            "time integration only. Use `adaptive=false` or a fixed-step " *
                            "algorithm, or pass `require_fixed_step=false` for experiments."))
    end

    set_iisph_projection_dt!(semi, integrator.dt)

    return integrator
end

function initialize_iisph_step_end_projection!(semi, project_at_step_end)
    uses_iisph_projection_dt(semi) || return semi

    if project_at_step_end
        enable_iisph_step_end_projection!(semi)
    else
        disable_iisph_step_end_projection!(semi)
    end

    return semi
end

function initialize_iisph_pressure_warm_start!(semi, warm_start_pressure)
    warm_start_pressure || return semi
    uses_iisph_projection_dt(semi) || return semi

    enable_iisph_pressure_warm_start!(semi)
    reset_iisph_pressure_initialization!(semi)

    return semi
end

function reset_iisph_pressure_warm_start!(semi, warm_start_pressure)
    warm_start_pressure || return semi
    uses_iisph_projection_dt(semi) || return semi

    reset_iisph_pressure_initialization!(semi)

    return semi
end

function update_iisph_pressure_warm_start!(semi, warm_start_pressure, integrator, t)
    warm_start_pressure || return semi
    uses_iisph_projection_dt(semi) || return semi
    iisph_pressure_warm_start_enabled(semi) || return semi

    return semi
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:IISPHTimeStepCallback})
    @nospecialize cb # reduce precompilation time

    callback = cb.affect!
    print(io, "IISPHTimeStepCallback(require_fixed_step=",
          callback.require_fixed_step, ", warm_start_pressure=",
          callback.warm_start_pressure, ", project_at_step_end=",
          callback.project_at_step_end, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:IISPHTimeStepCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        callback = cb.affect!

        setup = [
            "require fixed step" => string(callback.require_fixed_step),
            "warm-start pressure" => string(callback.warm_start_pressure),
            "project at step end" => string(callback.project_at_step_end)
        ]
        summary_box(io, "IISPHTimeStepCallback", setup)
    end
end

function Base.show(io::IO, limiter::IISPHTimeStepLimiter)
    @nospecialize limiter # reduce precompilation time

    print(io, "IISPHTimeStepLimiter(require_fixed_step=",
          limiter.require_fixed_step, ", warm_start_pressure=",
          limiter.warm_start_pressure, ")")
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:StepsizeCallback})
    @nospecialize cb # reduce precompilation time

    stepsize_callback = cb.affect!
    print(io, "StepsizeCallback(is_constant=", is_constant(stepsize_callback),
          ", cfl_number=", stepsize_callback.cfl_number, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:StepsizeCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        stepsize_callback = cb.affect!

        setup = [
            "is constant" => string(is_constant(stepsize_callback)),
            "CFL number" => stepsize_callback.cfl_number
        ]
        summary_box(io, "StepsizeCallback", setup)
    end
end

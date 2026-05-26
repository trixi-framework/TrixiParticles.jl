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

struct IISPHTimeStepCallback{R, W, P, M}
    require_fixed_step::R
    warm_start_pressure::W
    project_at_step_end::P
    pressure_projection::M
end

@doc raw"""
    IISPHTimeStepCallback(; require_fixed_step=true, warm_start_pressure=true,
                          project_at_step_end=false, pressure_projection=nothing)

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

`pressure_projection` selects how IISPH pressure is coupled to the time integrator:
- `:stage`: solve pressure inside every RHS evaluation. This is the default.
- `:step_end`: evaluate RK stages with the non-pressure RHS and project once after each
  accepted step.
- `:strang`: evaluate RK stages with the non-pressure RHS and use symmetric half-step
  pressure projections around the RK step.

For backwards compatibility, `project_at_step_end=true` selects `:step_end` unless
`pressure_projection` is passed explicitly.
"""
function IISPHTimeStepCallback(; require_fixed_step=true, warm_start_pressure=true,
                               project_at_step_end=false, pressure_projection=nothing)
    pressure_projection_ = normalize_iisph_pressure_projection(project_at_step_end,
                                                               pressure_projection)
    project_at_step_end_ = pressure_projection_ != :stage

    callback = IISPHTimeStepCallback{typeof(require_fixed_step),
                                     typeof(warm_start_pressure),
                                     typeof(project_at_step_end_),
                                     typeof(pressure_projection_)}(require_fixed_step,
                                                                   warm_start_pressure,
                                                                   project_at_step_end_,
                                                                   pressure_projection_)

    return DiscreteCallback(callback, callback;
                            initialize=initialize_iisph_time_step_callback,
                            save_positions=(false, false))
end

function normalize_iisph_pressure_projection(project_at_step_end, pressure_projection)
    pressure_projection_ = if isnothing(pressure_projection)
        project_at_step_end ? :step_end : :stage
    else
        pressure_projection
    end

    if !(pressure_projection_ in (:stage, :step_end, :strang))
        throw(ArgumentError("`pressure_projection` must be one of `:stage`, " *
                            "`:step_end`, or `:strang`"))
    end

    return pressure_projection_
end

function initialize_iisph_time_step_callback(discrete_callback, u, t, integrator)
    callback = discrete_callback.affect!
    initialize_iisph_pressure_warm_start!(integrator.p.semi,
                                          callback.warm_start_pressure)
    initialize_iisph_pressure_projection!(integrator.p.semi,
                                          callback.pressure_projection)
    sync_iisph_projection_dt!(integrator;
                              require_fixed_step=callback.require_fixed_step)

    if callback.pressure_projection == :strang && hasproperty(integrator, :u)
        project_iisph_pressure_at_step_end!(integrator; dt_factor=0.5)
    end

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

    if callback.pressure_projection == :step_end
        project_iisph_pressure_at_step_end!(integrator)
        reset_iisph_pressure_warm_start!(integrator.p.semi,
                                         callback.warm_start_pressure)
    elseif callback.pressure_projection == :strang
        dt_factor = iisph_final_time_reached(integrator) ? 0.5 : 1
        project_iisph_pressure_at_step_end!(integrator; dt_factor)
        reset_iisph_pressure_warm_start!(integrator.p.semi,
                                         callback.warm_start_pressure)
    else
        reset_iisph_pressure_warm_start!(integrator.p.semi,
                                         callback.warm_start_pressure)
    end

    # Step-end and Strang projection modes mutate the accepted state.
    u_modified!(integrator, callback.project_at_step_end)

    return callback
end

function iisph_final_time_reached(integrator)
    t_end = last(integrator.sol.prob.tspan)
    tolerance = 100 * eps(float(one(t_end))) * max(abs(t_end), one(abs(t_end)))

    if integrator.tdir > 0
        return integrator.t >= t_end - tolerance
    else
        return integrator.t <= t_end + tolerance
    end
end

struct IISPHTimeStepLimiter{R, W}
    require_fixed_step::R
    warm_start_pressure::W
end

struct IISPHPressureAdaptiveTimeStepCallback{MIN, MAX, TMIN, TMAX, G, S, C, R}
    min_dt::MIN
    max_dt::MAX
    target_min_iterations::TMIN
    target_max_iterations::TMAX
    growth_factor::G
    shrink_factor::S
    cap_shrink_factor::C
    require_fixed_step::R
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

@doc raw"""
    IISPHPressureAdaptiveTimeStepCallback(; min_dt=0, max_dt=Inf,
                                          target_iterations=(3, 8),
                                          growth_factor=1.25,
                                          shrink_factor=0.8,
                                          cap_shrink_factor=0.5,
                                          require_fixed_step=false)

Adapt the next accepted IISPH step size from pressure solver work.

This callback does not reject or roll back steps. It reads the IISPH pressure solver
iterations accumulated during the accepted step, proposes a new `dt` for the next step,
and resets the per-step pressure counters. This makes it suitable for IISPH experiments
where rejected steps would require restoring pressure caches.

Place this callback after [`IISPHTimeStepCallback`](@ref) in a `CallbackSet`, so
step-end and Strang pressure projections are included in the measured pressure work.
"""
function IISPHPressureAdaptiveTimeStepCallback(; min_dt=0, max_dt=Inf,
                                               target_iterations=(3, 8),
                                               growth_factor=1.25,
                                               shrink_factor=0.8,
                                               cap_shrink_factor=0.5,
                                               require_fixed_step=false)
    target_min, target_max = target_iterations

    min_dt < 0 && throw(ArgumentError("`min_dt` must be non-negative"))
    max_dt <= 0 && throw(ArgumentError("`max_dt` must be positive"))
    min_dt > max_dt && throw(ArgumentError("`min_dt` must be smaller than `max_dt`"))
    target_min <= 0 && throw(ArgumentError("target iteration counts must be positive"))
    target_min > target_max &&
        throw(ArgumentError("target iteration lower bound must not exceed upper bound"))
    growth_factor < 1 &&
        throw(ArgumentError("`growth_factor` must be greater than or equal to one"))
    !(0 < shrink_factor <= 1) &&
        throw(ArgumentError("`shrink_factor` must be in the interval (0, 1]"))
    !(0 < cap_shrink_factor <= 1) &&
        throw(ArgumentError("`cap_shrink_factor` must be in the interval (0, 1]"))

    callback = IISPHPressureAdaptiveTimeStepCallback{typeof(min_dt), typeof(max_dt),
                                                     typeof(target_min),
                                                     typeof(target_max),
                                                     typeof(growth_factor),
                                                     typeof(shrink_factor),
                                                     typeof(cap_shrink_factor),
                                                     typeof(require_fixed_step)}(min_dt,
                                                                                 max_dt,
                                                                                 target_min,
                                                                                 target_max,
                                                                                 growth_factor,
                                                                                 shrink_factor,
                                                                                 cap_shrink_factor,
                                                                                 require_fixed_step)

    return DiscreteCallback(callback, callback;
                            initialize=initialize_iisph_pressure_adaptive_callback,
                            save_positions=(false, false))
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

function initialize_iisph_pressure_adaptive_callback(discrete_callback, u, t, integrator)
    callback = discrete_callback.affect!
    sync_iisph_projection_dt!(integrator;
                              require_fixed_step=callback.require_fixed_step)
    reset_iisph_pressure_step_stats!(integrator.p.semi)

    return discrete_callback
end

# `condition`
function (callback::IISPHPressureAdaptiveTimeStepCallback)(u, t, integrator)
    return uses_iisph_projection_dt(integrator.p.semi)
end

# `affect!`
function (callback::IISPHPressureAdaptiveTimeStepCallback)(integrator)
    semi = integrator.p.semi
    stats = iisph_pressure_step_stats(semi)
    if stats.solve_count == 0
        u_modified!(integrator, false)
        return callback
    end

    pressure_max_iterations = minimum(maximum_iisph_iterations, semi.systems)
    dt = integrator.dt
    dt_new = iisph_pressure_adaptive_dt(dt, stats;
                                        min_dt=callback.min_dt,
                                        max_dt=callback.max_dt,
                                        target_min_iterations=callback.target_min_iterations,
                                        target_max_iterations=callback.target_max_iterations,
                                        growth_factor=callback.growth_factor,
                                        shrink_factor=callback.shrink_factor,
                                        cap_shrink_factor=callback.cap_shrink_factor,
                                        pressure_max_iterations)

    if dt_new != dt
        set_proposed_dt!(integrator, dt_new)
        integrator.dtcache = dt_new
        set_iisph_projection_dt!(semi, dt_new)
    end
    reset_iisph_pressure_step_stats!(semi)
    u_modified!(integrator, iisph_step_end_projection_enabled(semi))

    return callback
end

function iisph_pressure_adaptive_dt(dt, stats; min_dt=0, max_dt=Inf,
                                    target_min_iterations=3,
                                    target_max_iterations=8,
                                    growth_factor=1.25,
                                    shrink_factor=0.8,
                                    cap_shrink_factor=0.5,
                                    pressure_max_iterations=typemax(Int))
    average_iterations = stats.average_iterations
    factor = one(dt)

    if stats.max_iterations >= pressure_max_iterations
        factor = min(factor, cap_shrink_factor)
    end

    if average_iterations > target_max_iterations
        factor = min(factor, shrink_factor,
                     sqrt(target_max_iterations / average_iterations))
        factor = max(factor, cap_shrink_factor)
    elseif average_iterations < target_min_iterations
        safe_average = max(average_iterations, eps(float(average_iterations)))
        factor = min(growth_factor, sqrt(target_min_iterations / safe_average))
    end

    return clamp(dt * factor, min_dt, max_dt)
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

function initialize_iisph_pressure_projection!(semi, pressure_projection)
    uses_iisph_projection_dt(semi) || return semi

    if pressure_projection == :strang
        enable_iisph_strang_projection!(semi)
    elseif pressure_projection == :step_end
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
          callback.project_at_step_end, ", pressure_projection=",
          callback.pressure_projection, ")")
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
            "project at step end" => string(callback.project_at_step_end),
            "pressure projection" => string(callback.pressure_projection)
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

function Base.show(io::IO,
                   cb::DiscreteCallback{<:Any, <:IISPHPressureAdaptiveTimeStepCallback})
    @nospecialize cb # reduce precompilation time

    callback = cb.affect!
    print(io, "IISPHPressureAdaptiveTimeStepCallback(min_dt=",
          callback.min_dt, ", max_dt=", callback.max_dt,
          ", target_iterations=(", callback.target_min_iterations, ", ",
          callback.target_max_iterations, "), growth_factor=", callback.growth_factor,
          ", shrink_factor=", callback.shrink_factor,
          ", cap_shrink_factor=", callback.cap_shrink_factor, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:IISPHPressureAdaptiveTimeStepCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        callback = cb.affect!

        setup = [
            "minimum dt" => string(callback.min_dt),
            "maximum dt" => string(callback.max_dt),
            "target iterations" => string((callback.target_min_iterations,
                                            callback.target_max_iterations)),
            "growth factor" => string(callback.growth_factor),
            "shrink factor" => string(callback.shrink_factor),
            "cap shrink factor" => string(callback.cap_shrink_factor)
        ]
        summary_box(io, "IISPHPressureAdaptiveTimeStepCallback", setup)
    end
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

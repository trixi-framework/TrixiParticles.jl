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
    semi = integrator.p

    # If a `SplitIntegrationCallback` appears AFTER this `StepsizeCallback` in the
    # `CallbackSet`, then `semi.integrate_tlsph[]` has not yet been set to `false`.
    # In that situation, we cannot rely on `semi.integrate_tlsph[]`.
    # Instead, we must detect whether the list of callbacks contains
    # a `SplitIntegrationCallback`, and, if so, assume `integrate_tlsph = false`.
    integrate_tlsph = !any(cb -> cb isa DiscreteCallback{SplitIntegrationCallback},
                           integrator.opts.callback.discrete_callbacks)

    dt = @trixi_timeit timer() "calculate dt" calculate_dt(v_ode, u_ode, cfl_number, semi,
                                                           integrate_tlsph)

    set_proposed_dt!(integrator, dt)
    integrator.opts.dtmax = dt
    integrator.dtcache = dt

    # Tell OrdinaryDiffEq that `u` has not been modified
    u_modified!(integrator, false)

    return stepsize_callback
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

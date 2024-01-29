struct StepsizeCallback{ISCONSTANT, ELTYPE}
    cfl_number::ELTYPE
end

@inline is_constant(::StepsizeCallback{ISCONSTANT}) where {ISCONSTANT} = ISCONSTANT

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

    dt = @trixi_timeit timer() "calculate dt" calculate_dt(v_ode, u_ode, cfl_number, semi)

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
          "cfl_number=", stepsize_callback.cfl_number, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:StepsizeCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        stepsize_callback = cb.affect!

        setup = [
            "is constant" => is_constant(stepsize_callback),
            "CFL number" => stepsize_callback.cfl_number,
        ]
        summary_box(io, "StepsizeCallback", setup)
    end
end

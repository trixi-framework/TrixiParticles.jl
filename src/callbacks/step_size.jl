mutable struct StepSizeCallback
    start_intervall::Float64
    callback_interval::Int
end


function StepSizeCallback(; callback_interval=0)
    step_size_callback = StepSizeCallback(0.0, callback_interval)

    DiscreteCallback(step_size_callback, step_size_callback,
                     save_positions=(false, false),
                     initialize=initialize_dt)
end


# condition
function (step_size_callback::StepSizeCallback)(u, t, integrator)
    @unpack callback_interval = step_size_callback

    return callback_interval == 0 ||
        integrator.destats.naccept % callback_interval == 0 || t==0.0
end

# affect!
function (step_size_callback::StepSizeCallback)(integrator)
    semi = integrator.p
    @unpack cache = semi
    cache.dt[1] = integrator.dt
    # Tell OrdinaryDiffEq that u has not been modified
    u_modified!(integrator, false)
    return nothing
end

function initialize_dt(discrete_callback, u, t, integrator)
    integrator.p.cache.dt[1] = integrator.dt
    return nothing
end


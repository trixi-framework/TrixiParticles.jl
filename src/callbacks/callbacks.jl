# Used by `SolutionSavingCallback` and `DensityReinitializationCallback`
get_iter(::Integer, integrator) = integrator.stats.naccept
function get_iter(dt::AbstractFloat, integrator)
    # Basically `(t - tspan[1]) / dt` as `Int`.
    Int(div(integrator.t - first(integrator.sol.prob.tspan), dt, RoundNearest))
end

# Used by `InfoCallback` and `PostProcessCallback`
@inline function isfinished(integrator)
    # Checking for floating point equality is OK here as `DifferentialEquations.jl`
    # sets the time exactly to the final time in the last iteration
    return integrator.t == last(integrator.sol.prob.tspan) ||
           isempty(integrator.opts.tstops) ||
           integrator.iter == integrator.opts.maxiters
end

@inline function condition_integrator_interval(integrator, interval;
                                               save_final_solution=true)
    # With error-based step size control, some steps can be rejected. Thus,
    #   `integrator.iter >= integrator.stats.naccept`
    #    (total #steps)       (#accepted steps)
    # We need to check the number of accepted steps since callbacks are not
    # activated after a rejected step.
    return interval > 0 && ((integrator.stats.naccept % interval == 0) ||
            (save_final_solution && isfinished(integrator)))
end

function get_dvdu(integrator)
    vu_ode = integrator.u
    t = integrator.t
    dvdu_ode = similar(vu_ode)

    dv_ode, du_ode = dvdu_ode.x
    v_ode, u_ode = vu_ode.x

    kick!(dv_ode, v_ode, u_ode, integrator.p, t)
    drift!(du_ode, v_ode, u_ode, integrator.p, t)

    return dvdu_ode
end

# The `UpdateCallback` sets `semi.update_callback_used[]` to `true`,
# the `SplitIntegrationCallback` sets `semi.integrate_tlsph[]` to `false`.
# Callbacks like the `SolutionSavingCallback` and `PostprocessCallback` that call
# the RHS require these to be set correctly.
# Additionally, the `StepsizeCallback` requires `semi.integrate_tlsph[]` to be set correctly.
# However, if one of these callbacks appears BEFORE the `UpdateCallback`
# or `SplitIntegrationCallback` in the `CallbackSet`, then the flags have not been set yet
# when the callback is initialized.
# This function checks for the presence of these callbacks and sets the flags accordingly.
function set_callbacks_used!(semi, integrator)
    UpdateCB = Union{DiscreteCallback{<:Any, <:UpdateCallback},
                     DiscreteCallback{<:Any, <:PeriodicCallbackAffect{<:UpdateCallback}}}
    update_callback_used = any(cb -> cb isa UpdateCB,
                               integrator.opts.callback.discrete_callbacks)
    semi.update_callback_used[] = update_callback_used

    integrate_tlsph = !any(cb -> cb isa DiscreteCallback{<:Any, <:SplitIntegrationCallback},
                           integrator.opts.callback.discrete_callbacks)

    semi.integrate_tlsph[] = integrate_tlsph

    return semi
end

include("info.jl")
include("solution_saving.jl")
include("density_reinit.jl")
include("post_process.jl")
include("stepsize.jl")
include("update.jl")
include("steady_state_reached.jl")
include("split_integration.jl")

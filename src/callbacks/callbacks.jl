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

include("info.jl")
include("solution_saving.jl")
include("density_reinit.jl")
include("post_process.jl")
include("stepsize.jl")
include("update.jl")
include("steady_state_reached.jl")
include("split_integration.jl")

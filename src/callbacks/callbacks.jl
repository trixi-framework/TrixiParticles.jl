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

include("info.jl")
include("solution_saving.jl")
include("density_reinit.jl")
include("postprocess.jl")

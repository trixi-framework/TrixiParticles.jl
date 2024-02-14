# Used by `SolutionSavingCallback` and `DensityReinitializationCallback`
get_iter(::Integer, integrator) = integrator.stats.naccept
function get_iter(dt::AbstractFloat, integrator)
    # Basically `(t - tspan[1]) / dt` as `Int`.
    Int(div(integrator.t - first(integrator.sol.prob.tspan), dt, RoundNearest))
end

include("info.jl")
include("solution_saving.jl")
include("density_reinit.jl")
include("stepsize.jl")

# helper function to reduce lines in tutorials
function default_callback(; save_dt=0.02, info_interval=100)
    return CallbackSet(InfoCallback(interval=info_interval),
                       SolutionSavingCallback(dt=save_dt, prefix=""))
end

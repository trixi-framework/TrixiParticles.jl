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

function validate_interval_and_dt(interval, dt)
    if interval == 0 && dt == 0.0
        throw(ArgumentError("Both interval and dt cannot be zero. Specify at least one."))
    end
    if interval < 0 || dt < 0
        throw(ArgumentError("Negative interval or dt values are not supported."))
    end
    if dt > 0 && interval > 0
        throw(ArgumentError("Setting both interval and dt is not supported."))
    end
end

mutable struct OutputConfig
    output_directory::String
    filename::String
    prefix::String
    append_timestamp::Bool
    verbose::Bool

    function OutputConfig(;
                          output_directory::String="out",
                          filename::String="output",
                          prefix::String="",
                          append_timestamp::Bool=false,
                          verbose::Bool=false)
        new(output_directory, filename, prefix, append_timestamp, verbose)
    end
end

function build_filepath(config::OutputConfig; extension::String="")
    # Prepare output directory
    output_directory = config.output_directory

    # Add timestamp subdirectory if required
    if config.append_timestamp
        timestamp = Dates.format(Dates.now(), "YY-mm-ddTHHMMSS")
        output_directory = joinpath(output_directory, timestamp)
        mkpath(output_directory)
    end

    # Combine prefix and filename
    base_filename = config.prefix * config.filename

    # Add file extension if provided
    if !isempty(extension)
        base_filename *= "." * extension
    end

    # Construct the full path
    return joinpath(output_directory, base_filename)
end

include("info.jl")
include("solution_saving.jl")
include("density_reinit.jl")
include("post_process.jl")
include("stepsize.jl")
include("update.jl")
include("steady_state_reached.jl")

"""
    save_checkpoint(sol; output_directory="out_checkpoints", filename="checkpoint")

Save `sol` in a JLD2 file for simulation restart capabilities.

# Arguments
- `sol`:              The `ODESolution` returned by `solve` of `OrdinaryDiffEq`.
- `output_directory`: Directory to save the JLD2 file. Defaults to `"out_checkpoints"`.
- `filename`:         Name of the JLD2 file (without extension). Defaults to `"checkpoint"`.

# Returns
- `file::String`: Path to the saved checkpoint file.
"""
function save_checkpoint(sol::TrixiParticlesODESolution;
                         output_directory="out_checkpoints", filename="checkpoint")
    isdir(output_directory) || mkpath(output_directory)

    file = joinpath(output_directory, filename * ".jld2")

    JLD2.jldopen(file, "w") do f
        f["sol"] = sol
    end

    return file
end

"""
    load_checkpoint(file)

Load checkpoint data from a JLD2 file.

# Arguments
- `file::String`: Path to the checkpoint JLD2 file.

# Returns
- `sol::ODESolution`: The `ODESolution` loaded from the checkpoint file.
"""
function load_checkpoint(file)
    return JLD2.load(file)["sol"]
end

"""
    semidiscretize_from_checkpoint(sol::ODESolution, tspan)

Create a new `ODEProblem` from a checkpoint, restarting the simulation with the loaded state.

# Arguments
- `sol::ODESolution`: The solution loaded from a checkpoint (see [`load_checkpoint`](@ref)).
- `tspan::Tuple`: Time span for the continued simulation.

# Returns
- `ode:ODEProblem`: A new `ODEProblem` object initialized from the checkpoint state.

# See Also
- [`load_checkpoint`](@ref): Load a checkpoint from a file.
- [`save_checkpoint`](@ref): Save a checkpoint to a file.
"""
function semidiscretize_from_checkpoint(sol::TrixiParticlesODESolution, tspan)
    restart_with!(sol)

    return semidiscretize(sol.prob.p, tspan)
end

function restart_with!(sol::TrixiParticlesODESolution)
    return restart_with!(sol.prob.p, sol)
end

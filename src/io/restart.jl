"""
    save_checkpoint(semi, sol; output_directory="out", filename="solution")

Save `semi` and `sol` in a JLD2 file for simulation restart capabilities.

# Arguments
- `semi`:             The semidiscretization.
- `sol`:              The `ODESolution` returned by `solve` of `OrdinaryDiffEq`.
- `output_directory`: Directory to save the JLD2 file.
- `filename`:         Name of the JLD2 file.

# Example
# See "examples/postprocessing/restart_moving_wall_2d.jl"
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
    load_checkpoint(t_end; input_directory="out", filename="solution")

Load the `ode` from a JLD2 file for simulation restart.

# Arguments
- `t_end`:            The final time for the continuation of the simulation.
- `input_directory`:  Directory where the JLD2 file is located.
- `filename`:         Name of the JLD2 file.

# Returns
- `ode`: The new `ODEProblem` object, initialized with the state and setup loaded
         from the file.

# Example
# See "examples/postprocessing/restart_moving_wall_2d.jl"

"""
function load_checkpoint(file)
    return JLD2.load(file)["sol"]
end

function semidiscretize_from_checkpoint(sol::TrixiParticlesODESolution, tspan)
    restart_with!(sol)

    return semidiscretize(sol.prob.p, tspan)
end

function restart_with!(sol::TrixiParticlesODESolution)
    return restart_with!(sol.prob.p, sol)
end

"""
    save_jld2(semi, sol; output_directory="out", filename="solution")

Save `semi` and `sol` in a JLD2 file for simulation restart capabilities.

# Arguments
- `semi`:             The semidiscretization.
- `sol`:              The `ODESolution` returned by `solve` of `OrdinaryDiffEq`.
- `output_directory`: Directory to save the JLD2 file.
- `filename`:         Name of the JLD2 file.

# Example
# See "examples/postprocessing/restart_moving_wall_2d.jl"
"""
function save_jld2(semi, sol; output_directory="out", filename="solution")
    mkpath(output_directory)

    file = joinpath(output_directory, filename*".jld2")

    JLD2.jldopen(file, "w") do f
        f["semi"] = semi
        f["sol"] = sol
    end

    return
end

"""
    load_jld2(t_end; input_directory="out", filename="solution")

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
function load_jld2(t_end; input_directory="out", filename="solution")
    if !isdir(input_directory)
        error("input_directory` \"$input_directory\" not found.")
    end
    file = joinpath(input_directory, filename*".jld2")

    data = JLD2.load(file)

    # Retrieve fields
    semi = data["semi"]
    sol = data["sol"]

    tspan = (sol.t[2], t_end)

    semi = restart_with!(semi, (u=[sol.u[2]],))

    ode = semidiscretize(semi, tspan)

    return ode
end

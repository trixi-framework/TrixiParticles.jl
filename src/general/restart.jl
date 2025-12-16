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
function semidiscretize_from_checkpoint(sol::TrixiParticlesODESolution, tspan;
                                        initialization_backend=PolyesterBackend())
    restart_with!(sol; initialization_backend)

    return semidiscretize(sol.prob.p, tspan; initialization_backend)
end

function restart_with!(sol::TrixiParticlesODESolution;
                       initialization_backend=PolyesterBackend())
    return restart_with!(sol.prob.p, sol; initialization_backend)
end

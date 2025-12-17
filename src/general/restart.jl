struct CheckpointSolution{V, U, ELTYPE}
    v_ode::V
    u_ode::U
    time_stamp::ELTYPE
    semi::Semidiscretization
end

function Base.show(io::IO, sol::CheckpointSolution)
    @nospecialize sol # reduce precompilation time

    print(io, "CheckpointSolution(")
    semi = sol.semi
    for system in semi.systems
        print(io, system, ", ")
    end
    print(io, "neighborhood_search=")
    print(io, semi.neighborhood_searches |> eltype |> eltype |> nameof)
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", sol::CheckpointSolution)
    @nospecialize sol # reduce precompilation time

    if get(io, :compact, false)
        show(io, sol)
    else
        semi = sol.semi
        summary_header(io, "CheckpointSolution")
        summary_line(io, "time stamp", "$(sol.time_stamp) sec.")
        summary_line(io, "#spatial dimensions", ndims(semi.systems[1]))
        summary_line(io, "#systems", length(semi.systems))
        summary_line(io, "neighborhood search",
                     semi.neighborhood_searches |> eltype |> eltype |> nameof)
        summary_line(io, "total #particles", sum(nparticles.(semi.systems)))
        summary_line(io, "eltype", eltype(semi.systems[1]))
        summary_line(io, "coordinates eltype", coordinates_eltype(semi.systems[1]))
        summary_footer(io)
    end
end

"""
    semidiscretize_from_checkpoint(sol::TrixiParticles.CheckpointSolution, tspan)

Create a new `ODEProblem` from a checkpoint, restarting the simulation with the loaded state.

# Arguments
- `sol::ODESolution`: The solution loaded from a checkpoint (see [`load_checkpoint`](@ref)).
- `tspan::Tuple`: Time span for the continued simulation.

# Returns
- `ode:ODEProblem`: A new `ODEProblem` object initialized from the checkpoint state.

# See Also
- [`load_checkpoint`](@ref): Load a checkpoint from a file.
- [`save_checkpoint`](@ref): Save a checkpoint to a file.

!!! note
    This is an experimental implementation.
    Currently, restart is not supported for TLSPH systems whose initial configuration
    moves in space (e.g., falling elastic spheres).
"""
function semidiscretize_from_checkpoint(sol::CheckpointSolution, tspan)
    (; semi, v_ode, u_ode) = sol

    if semi.parallelization_backend isa KernelAbstractions.Backend
        # We have a different `Semidiscretization` in `sol`.
        # This means that systems linking to other systems point to other systems.
        # Therefore, we have to re-link them, which yields yet another `Semidiscretization`.
        # Note that this re-creates systems containing links, so it only works as long
        # as systems don't link to other systems containing links.
        semi_new = Semidiscretization(set_system_links.(semi.systems, Ref(semi)),
                                      semi.ranges_u, semi.ranges_v,
                                      semi.neighborhood_searches,
                                      semi.parallelization_backend,
                                      semi.update_callback_used, semi.integrate_tlsph)
    else
        semi_new = semi
    end

    foreach_system(semi_new) do system
        v = wrap_v(v_ode, system, semi_new)
        u = wrap_u(u_ode, system, semi_new)

        restart_with!(system, v, u)
    end

    # Reset callback flag that will be set by the `UpdateCallback`
    semi_new.update_callback_used[] = false

    return DynamicalODEProblem(kick!, drift!, v_ode, u_ode, tspan, semi_new)
end

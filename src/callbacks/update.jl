struct UpdateEachTimeStep
    interval :: Int
    update   :: Bool
end

function UpdateEachTimeStep(; update=true, interval::Integer=1)
    update_each_dt! = UpdateEachTimeStep(interval, update)

    # Update each step
    condition_ = (u, t, integrator) -> update

    return DiscreteCallback(condition_, update_each_dt!, initialize=initial_update!,
                            save_positions=(false, false))
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:UpdateEachTimeStep})
    @nospecialize cb # reduce precompilation time
    print(io, "UpdateEachTimeStep()")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:UpdateEachTimeStep})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        update_each_dt = cb.affect!
        setup = [
            "interval" => update_each_dt.interval,
            "update" => update_each_dt.update ? "yes" : "no",
        ]
        summary_box(io, "UpdateEachTimeStep", setup)
    end
end

# initialize
initial_update!(cb, u, t, integrator) = cb.affect!(integrator)

# condition
function (update_each_dt!::UpdateEachTimeStep)(u, t, integrator)
    (; interval, update) = solution_callback

    # With error-based step size control, some steps can be rejected. Thus,
    #   `integrator.iter >= integrator.stats.naccept`
    #    (total #steps)       (#accepted steps)
    # We need to check the number of accepted steps since callbacks are not
    # activated after a rejected step.
    return update && integrator.stats.naccept % interval == 0
end

# affect
function (update_each_dt!::UpdateEachTimeStep)(integrator)
    semi = integrator.p
    v_ode, u_ode = integrator.u.x

    foreach_system(semi) do system
        update_open_boundary_eachstep!(system, v_ode, u_ode, semi, integrator.t)
    end

    # Tell OrdinaryDiffEq that u has been modified
    u_modified!(integrator, true)

    return integrator
end

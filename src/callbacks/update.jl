# TODO: naming
struct UpdateEachDt end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:UpdateEachDt})
    @nospecialize cb # reduce precompilation time
    print(io, "UpdateEachTimeStep()")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:UpdateEachDt})
    @nospecialize cb # reduce precompilation time
    if get(io, :compact, false)
        show(io, cb)
    else
        summary_box(io, "UpdateEachTimeStep")
    end
end

function UpdateEachTimeStep()
    update_each_dt = UpdateEachDt()
    return DiscreteCallback(update_each_dt, update_each_dt, initialize=initial_update!,
                            save_positions=(false, false))
end

# condition
(update_each_dt::UpdateEachDt)(u, t, integrator) = true

# affect
function (update_each_dt::UpdateEachDt)(integrator)
    semi = integrator.p
    v_ode, u_ode = integrator.u.x

    foreach_enumerate(semi.systems) do (system_index, system)
        update_open_boundary_eachstep!(system, system_index, v_ode, u_ode, semi)
    end

    foreach_enumerate(semi.systems) do (system_index, system)
        update_transport_velocity!(system, system_index, v_ode, u_ode, semi)
    end

    # Tell OrdinaryDiffEq that u has been modified
    u_modified!(integrator, true)

    return integrator
end

# initialize
initial_update!(cb, u, t, integrator) = cb.affect!(integrator)

# TODO: naming
struct UpdateAfterDt end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:UpdateAfterDt})
    @nospecialize cb # reduce precompilation time
    print(io, "UpdateAfterTimeStep()")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:UpdateAfterDt})
    @nospecialize cb # reduce precompilation time
    if get(io, :compact, false)
        show(io, cb)
    else
        summary_box(io, "UpdateAfterTimeStep")
    end
end

function UpdateAfterTimeStep()

    update_after_dt = UpdateAfterDt()
    return DiscreteCallback(update_after_dt, update_after_dt, initialize=initial_update!,
                            save_positions=(false, false))
end

# condition
(update_after_dt::UpdateAfterDt)(u, t, integrator) = true

# affect
function (update_after_dt::UpdateAfterDt)(integrator)
    semi = integrator.p
    v_ode, u_ode = integrator.u.x

    foreach_enumerate(semi.systems) do (system_index, system)
        update_open_boundary_eachstep!(system, system_index, v_ode, u_ode, semi)
    end

    foreach_enumerate(semi.systems) do (system_index, system)
        update_transport_velocity!(system, system_index, v_ode, u_ode, semi)
    end

    return integrator
end

# initialize
initial_update!(cb, u, t, integrator) = cb.affect!

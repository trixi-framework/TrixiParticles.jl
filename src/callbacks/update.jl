# TODO: naming
function UpdateAfterTimeStep()
    return DiscreteCallback(update_condition, update_after_dt!, initialize=initial_update!,
                            save_positions=(false, false))
end

# condition
update_condition(u, t, integrator) = true

# affect
function update_after_dt!(integrator)
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
initial_update!(cb, u, t, integrator) = update_after_dt!(integrator)

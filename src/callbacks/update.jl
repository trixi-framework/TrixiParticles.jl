# TODO: naming
function UpdateAfterTimeStep()
    return DiscreteCallback(update_condition, update_after_dt!, initialize=initial_update!)
end

# condition
update_condition(u, t, integrator) = true

# affect
function update_after_dt!(integrator)
    semi = integrator.p
    v_ode, u_ode = integrator.u.x

    foreach_enumerate(semi.systems) do (system_index, system)
        update_average_pressure!(system, system_index, v_ode, u_ode, semi)
        update_transport_velocity!(system, system_index, v_ode, u_ode, semi)
    end

    return integrator
end

# initialize
function initial_update!(cb, u, t, integrator)
    semi = integrator.p
    v_ode, u_ode = integrator.u.x

    update_systems_and_nhs(v_ode, u_ode, semi, t)

    update_after_dt!(integrator)
end

update_average_pressure!(system, system_index, v_ode, u_ode, semi) = system
update_transport_velocity!(system, system_index, v_ode, u_ode, semi) = system

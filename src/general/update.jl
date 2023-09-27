function update_step!(u, ode, semi, t)
    v_ode, u_ode = ode.u.x

    foreach_enumerate(semi.systems) do (system_index, system)
        update_open_boundary_eachstep!(system, system_index, v_ode, u_ode, semi)
    end

    foreach_enumerate(semi.systems) do (system_index, system)
        update_transport_velocity!(system, system_index, v_ode, u_ode, semi)
    end

    return ode
end

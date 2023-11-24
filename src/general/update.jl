function update_step!(u, ode, semi, t)
    v_ode, u_ode = ode.u.x

    foreach_system(semi) do system
        update_open_boundary_eachstep!(system, v_ode, u_ode, semi)
    end

    foreach_system(semi) do system
        update_transport_velocity!(system, v_ode, semi)
    end

    return ode
end

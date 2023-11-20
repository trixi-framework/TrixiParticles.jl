struct TransportVelocityAdami{ELTYPE}
    background_pressure::ELTYPE
    function TransportVelocityAdami(background_pressure)
        new{typeof(background_pressure)}(background_pressure)
    end
end

function update_transport_velocity!(u, ode, semi, t)
    v_ode, u_ode = ode.u.x

    foreach_system(semi) do system
        update_transport_velocity!(system, v_ode, semi)
    end

    return ode
end

@inline update_transport_velocity!(system, v_ode, semi) = system

@inline function update_transport_velocity!(system::FluidSystem, v_ode, semi)
    update_transport_velocity!(system, v_ode, semi, system.transport_velocity)
end

@inline update_transport_velocity!(system, v_ode, semi, ::Nothing) = system

@inline function update_transport_velocity!(system, v_ode, semi, ::TransportVelocityAdami)
    (; cache) = system
    v = wrap_v(v_ode, system, semi)

    for particle in each_moving_particle(system)
        for i in 1:ndims(system)
            cache.advection_velocity[i, particle] = v[ndims(system) + i, particle]
            v[ndims(system) + i, particle] = v[i, particle]
        end
    end
end

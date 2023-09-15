struct TransportVelocityAdami{ELTYPE}
    background_pressure::ELTYPE
    function TransportVelocityAdami(background_pressure)
        new{typeof(background_pressure)}(background_pressure)
    end
end

@inline function update_transport_velocity!(system, system_index, v_ode, u_ode, semi)
    return system
end

@inline function update_transport_velocity!(system::FluidSystem, system_index,
                                            v_ode, u_ode, semi)
    update_transport_velocity!(system, system_index, v_ode, u_ode, semi,
                               system.transport_velocity)
end

@inline function update_transport_velocity!(system, system_index, v_ode, u_ode, semi,
                                            ::Nothing)
    return system
end

@inline function update_transport_velocity!(system, system_index, v_ode, u_ode, semi,
                                            ::TransportVelocityAdami)
    (; cache) = system
    v = wrap_v(v_ode, system_index, system, semi)

    for particle in each_moving_particle(system)
        for i in 1:ndims(system)
            cache.advection_velocity[i, particle] = v[ndims(system) + i, particle]
            v[ndims(system) + i, particle] = v[i, particle]
        end
    end
end

function set_transport_velocity!(system::FluidSystem,
                                 particle, particle_old, v, v_old)
    set_transport_velocity!(system, particle, particle_old, v, v_old,
                            system.transport_velocity)
end

set_transport_velocity!(system, particle, particle_old, v, v_old) = system

set_transport_velocity!(system, particle, particle_old, v, v_old, ::Nothing) = system

function set_transport_velocity!(system, particle, particle_old, v, v_old,
                                 ::TransportVelocityAdami)
    for i in 1:ndims(system)
        system.cache.advection_velocity[i, particle] = v_old[i, particle_old]
        v[ndims(system) + i, particle] = v_old[i, particle_old]
    end
end

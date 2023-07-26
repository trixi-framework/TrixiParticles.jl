struct TransportVelocityAdami{ELTYPE}
    background_pressure::ELTYPE
    function TransportVelocityAdami(background_pressure)
        new{typeof(background_pressure)}(background_pressure)
    end
end

# TODO
@inline function average_pressure(system, particle)
    return 0.0
end

@inline function update_transport_velocity!(system::FluidSystem,
                                            system_index, v_ode, u_ode, semi)
    update_transport_velocity!(system, system_index, v_ode, u_ode, semi,
                               system.transport_velocity)
end

@inline function update_transport_velocity!(system, system_index, v_ode, u_ode, semi,
                                            ::TransportVelocityAdami)
    v = wrap_v(v_ode, system_index, system, semi)

    for particle in each_moving_particle(system)
        for i in 1:ndims(system)
            v[ndims(system) + i, particle] = v[i, particle]
        end
    end
end

@inline function update_transport_velocity!(system, system_index, v_ode, u_ode, semi,
                                            ::Nothing)
    return system
end

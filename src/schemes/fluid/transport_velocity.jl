struct TransportVelocityAdami{ELTYPE}
    background_pressure::ELTYPE
    function TransportVelocityAdami(background_pressure)
        new{typeof(background_pressure)}(background_pressure)
    end
end

@inline update_transport_velocity!(system, v_ode, semi) = system

@inline function update_transport_velocity!(system::FluidSystem, v_ode, semi)
    update_transport_velocity!(system, v_ode, semi, system.transport_velocity)
end

@inline update_transport_velocity!(system, v_ode, semi, ::Nothing) = system

@inline function update_transport_velocity!(system, v_ode, semi, ::TransportVelocityAdami)
    v = wrap_v(v_ode, system, semi)

    for particle in each_moving_particle(system)
        for i in 1:ndims(system)
            v[ndims(system) + i, particle] = v[i, particle]
        end
    end

    return system
end

# Add momentum velocity.
@inline function add_velocity!(du, v, particle, system, ::Nothing)
    for i in 1:ndims(system)
        du[i, particle] = v[i, particle]
    end

    return du
end

# Add advection velocity.
@inline function add_velocity!(du, v, particle, system, ::TransportVelocityAdami)
    for i in 1:ndims(system)
        du[i, particle] = v[ndims(system) + i, particle]
    end

    return du
end

@inline function advection_velocity(v, system, particle)
    return SVector(ntuple(@inline(dim->v[ndims(system) + dim, particle]), ndims(system)))
end

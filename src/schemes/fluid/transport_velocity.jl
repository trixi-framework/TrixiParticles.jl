struct TransportVelocityAdami{ELTYPE}
    background_pressure::ELTYPE
    function TransportVelocityAdami(background_pressure)
        new{typeof(background_pressure)}(background_pressure)
    end
end

# Calculate `v_nvariables` appropriately
@inline factor_tvf(system::FluidSystem) = factor_tvf(system, system.transport_velocity)
@inline factor_tvf(system, ::Nothing) = 1
@inline factor_tvf(system, ::TransportVelocityAdami) = 2

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

function write_v0!(v0, system::FluidSystem, ::TransportVelocityAdami)
    (; initial_condition) = system

    for particle in eachparticle(system)
        # Write particle velocities
        for dim in 1:ndims(system)
            v0[dim, particle] = initial_condition.velocity[dim, particle]
            v0[ndims(system) + dim, particle] = initial_condition.velocity[dim, particle]
        end
    end

    write_v0!(v0, system.density_calculator, system)

    return v0
end

@inline function add_velocity!(du, v, particle, system::FluidSystem)
    add_velocity!(du, v, particle, system, system.transport_velocity)
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

@inline function momentum_convection(system, neighbor_system,
                                     v_particle_system, v_neighbor_system, rho_a, rho_b,
                                     m_a, m_b, particle, neighbor, grad_kernel)
    return SVector(ntuple(_ -> 0.0, Val(ndims(system))))
end

@inline function momentum_convection(system::FluidSystem,
                                     neighbor_system::FluidSystem,
                                     v_particle_system, v_neighbor_system, rho_a, rho_b,
                                     m_a, m_b, particle, neighbor, grad_kernel)
    momentum_convection(system, neighbor_system, system.transport_velocity,
                        v_particle_system, v_neighbor_system, rho_a, rho_b,
                        m_a, m_b, particle, neighbor, grad_kernel)
end

@inline function momentum_convection(system, neighbor_system, ::Nothing,
                                     v_particle_system, v_neighbor_system, rho_a, rho_b,
                                     m_a, m_b, particle, neighbor, grad_kernel)
    return SVector(ntuple(_ -> 0.0, Val(ndims(system))))
end

@inline function momentum_convection(system, neighbor_system, ::TransportVelocityAdami,
                                     v_particle_system, v_neighbor_system, rho_a, rho_b,
                                     m_a, m_b, particle, neighbor, grad_kernel)
    volume_a = m_a / rho_a
    volume_b = m_b / rho_b
    volume_term = (volume_a^2 + volume_b^2) / m_a

    momentum_velocity_a = current_velocity(v_particle_system, system, particle)
    advection_velocity_a = advection_velocity(v_particle_system, system, particle)

    momentum_velocity_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)
    advection_velocity_b = advection_velocity(v_neighbor_system, neighbor_system, neighbor)

    A_a = rho_a * momentum_velocity_a * (advection_velocity_a - momentum_velocity_a)'
    A_b = rho_b * momentum_velocity_b * (advection_velocity_b - momentum_velocity_b)'

    return volume_term * (0.5 * (A_a + A_b)) * grad_kernel
end

@inline transport_velocity!(dv, system, rho_a, rho_b, m_a, m_b, grad_kernel, particle) = dv

@inline function transport_velocity!(dv, system::FluidSystem,
                                     rho_a, rho_b, m_a, m_b, grad_kernel, particle)
    transport_velocity!(dv, system, system.transport_velocity, rho_a, rho_b, m_a, m_b,
                        grad_kernel, particle)
end

@inline function transport_velocity!(dv, system, ::Nothing,
                                     rho_a, rho_b, m_a, m_b, grad_kernel, particle)
    return dv
end

@inline function transport_velocity!(dv, system, ::TransportVelocityAdami,
                                     rho_a, rho_b, m_a, m_b, grad_kernel, particle)
    (; transport_velocity) = system
    (; background_pressure) = transport_velocity

    NDIMS = ndims(system)

    volume_a = m_a / rho_a
    volume_b = m_b / rho_b
    volume_term = (volume_a^2 + volume_b^2) / m_a

    for dim in 1:NDIMS
        dv[NDIMS + dim, particle] -= volume_term * background_pressure * grad_kernel[dim]
    end

    return dv
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

"""
    TransportVelocityAdami(background_pressure::Real)

Transport Velocity Formulation (TVF) to suppress pairing and tensile instability.
See [TVF](@ref transport_velocity_formulation) for more details of the method.

# Arguments
- `background_pressure`: Background pressure. Suggested is a background pressure which is
                         on the order of the reference pressure.

!!! note
    There is no need for an artificial viscosity to suppress tensile instability when using `TransportVelocityAdami`.
    Thus, it is highly recommended to use [`ViscosityAdami`](@ref) as viscosity model,
    since [`ArtificialViscosityMonaghan`](@ref) leads to bad results.
"""
struct TransportVelocityAdami{T <: Real}
    background_pressure::T
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
    @threaded semi for particle in each_moving_particle(system)
        for i in 1:ndims(system)
            v[ndims(system) + i, particle] = v[i, particle]
        end
    end

    return system
end

function write_v0!(v0, system::FluidSystem, ::TransportVelocityAdami)
    (; initial_condition) = system

    # Write particle velocities
    for dim in 1:ndims(system)
        v0[ndims(system) + dim, :] = initial_condition.velocity[dim, :]
    end

    return v0
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

@inline function momentum_convection(system, neighbor_system, ::Nothing,
                                     v_particle_system, v_neighbor_system, rho_a, rho_b,
                                     m_a, m_b, particle, neighbor, grad_kernel)
    return zero(grad_kernel)
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

    return volume_term * ((A_a + A_b) / 2) * grad_kernel
end

@inline transport_velocity!(dv, system, neighbor_system, particle, neighbor, m_a, m_b,
                            grad_kernel) = dv

@inline function transport_velocity!(dv, system::FluidSystem, neighbor_system,
                                     particle, neighbor, m_a, m_b, grad_kernel)
    transport_velocity!(dv, system, neighbor_system, system.transport_velocity,
                        particle, neighbor, m_a, m_b, grad_kernel)
end

@inline function transport_velocity!(dv, system, neighbor_system, ::Nothing,
                                     particle, neighbor, m_a, m_b, grad_kernel)
    return dv
end

@inline function transport_velocity!(dv, system, neighbor_system, ::TransportVelocityAdami,
                                     particle, neighbor, m_a, m_b, grad_kernel)
    (; transport_velocity) = system
    (; background_pressure) = transport_velocity

    NDIMS = ndims(system)

    # The TVF is based on the assumption that the pressure gradient is only accurately
    # computed when the particle distribution is isotropic.
    # That means, the force contribution vanishes only if the particle distribution is
    # isotropic AND the field being differentiated by the kernel gradient is spatially constant.
    # So we must guarantee a constant field and therefore the reference density is used
    # instead of the locally computed one.
    # TODO:
    # volume_a = particle_spacing(system, particle)^ndims(system)
    # volume_b = particle_spacing(neighbor_system, neighbor)^ndims(neighbor_system)
    volume_a = m_a / system.initial_condition.density[particle]
    volume_b = m_b / neighbor_system.initial_condition.density[neighbor]

    volume_term = (volume_a^2 + volume_b^2) / m_a

    for dim in 1:NDIMS
        dv[NDIMS + dim, particle] -= volume_term * background_pressure * grad_kernel[dim]
    end

    return dv
end

function reset_callback_flag!(system::FluidSystem)
    reset_callback_flag!(system, system.transport_velocity)
end

reset_callback_flag!(system, ::Nothing) = system

function reset_callback_flag!(system::FluidSystem, ::TransportVelocityAdami)
    system.cache.update_callback_used[] = false

    return system
end

function update_callback_used!(system::FluidSystem)
    update_callback_used!(system, system.transport_velocity)
end

update_callback_used!(system, ::Nothing) = system

function update_callback_used!(system, transport_velocity)
    system.cache.update_callback_used[] = true
end

function check_tvf_configuration(system::FluidSystem, ::Nothing,
                                 v, u, v_ode, u_ode, semi, t; update_from_callback=false)
    return system
end

function check_tvf_configuration(system::FluidSystem, ::TransportVelocityAdami,
                                 v, u, v_ode, u_ode, semi, t; update_from_callback=false)
    # The `UpdateCallback` will set `system.cache.update_callback_used[]` to `true`
    # in the initialization. However, other callbacks might update the system first, hence `update_from_callback`.
    if !update_from_callback && !(system.cache.update_callback_used[])
        throw(ArgumentError("`UpdateCallback` is required when using `TransportVelocityAdami`"))
    end

    return system
end

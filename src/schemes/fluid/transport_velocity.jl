"""
    TransportVelocityAdami(background_pressure::Real)

Transport Velocity Formulation (TVF) to suppress pairing and tensile instability.
See [TVF](@ref transport_velocity_formulation) for more details of the method.

# Arguments
- `background_pressure`: Background pressure. Suggested is a background pressure which is
                         on the order of the reference pressure.
"""
struct TransportVelocityAdami{T <: Real}
    background_pressure::T
end

# No TVF for a system by default
@inline transport_velocity(system) = nothing

# `δv` is the correction to the particle velocity due to the TVF.
# Particles are advected with the velocity `v + δv`.
@propagate_inbounds function delta_v(system, particle)
    return delta_v(system, transport_velocity(system), particle)
end

@propagate_inbounds function delta_v(system, ::TransportVelocityAdami, particle)
    return extract_svector(system.cache.delta_v, system, particle)
end

# Zero when no TVF is used
@inline function delta_v(system, transport_velocity, particle)
    return zero(SVector{ndims(system), eltype(system)})
end

@inline function dv_transport_velocity(::Nothing, system, neighbor_system,
                                       v_particle_system, v_neighbor_system, rho_a, rho_b,
                                       m_a, m_b, particle, neighbor, grad_kernel)
    return zero(grad_kernel)
end

# @inline function dv_transport_velocity(::TransportVelocityAdami, system, neighbor_system,
#                                        v_particle_system, v_neighbor_system, rho_a, rho_b,
#                                        m_a, m_b, particle, neighbor, grad_kernel)
#     v_a = current_velocity(v_particle_system, system, particle)
#     delta_v_a = delta_v(system, particle)

#     v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)
#     delta_v_b = delta_v(neighbor_system, neighbor)

#     A_a = rho_a * v_a * delta_v_a'
#     A_b = rho_b * v_b * delta_v_b'

#     return m_b * ((A_a / rho_a^2 + A_b / rho_b^2) / 2) * grad_kernel
# end

@inline function dv_transport_velocity(::TransportVelocityAdami, system, neighbor_system,
                                       v_particle_system, v_neighbor_system, rho_a, rho_b,
                                       m_a, m_b, particle, neighbor, grad_kernel)
    volume_a = m_a / rho_a
    volume_b = m_b / rho_b
    volume_term = (volume_a^2 + volume_b^2) / m_a

    v_a = current_velocity(v_particle_system, system, particle)
    delta_v_a = delta_v(system, particle)

    v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)
    delta_v_b = delta_v(neighbor_system, neighbor)

    A_a = rho_a * v_a * delta_v_a'
    A_b = rho_b * v_b * delta_v_b'

    return volume_term * ((A_a + A_b) / 2) * grad_kernel
end

function update_tvf!(system, transport_velocity, v, u, v_ode, u_ode, semi, t)
    return system
end

function update_tvf!(system, transport_velocity::TransportVelocityAdami, v, u, v_ode,
                     u_ode, semi, t)
    (; delta_v) = system.cache
    (; background_pressure) = transport_velocity

    set_zero!(delta_v)

    foreach_system(semi) do neighbor_system
        u_neighbor = wrap_u(u_ode, neighbor_system, semi)

        system_coords = current_coordinates(u, system)
        neighbor_coords = current_coordinates(u_neighbor, neighbor_system)

        foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords,
                               semi;
                               points=each_moving_particle(system)) do particle, neighbor,
                                                                       pos_diff, distance
            m_a = hydrodynamic_mass(neighbor_system, neighbor)
            m_b = hydrodynamic_mass(neighbor_system, neighbor)

            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)

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

            delta_v_ = -volume_term * background_pressure * grad_kernel

            # Write into the buffer
            for i in eachindex(delta_v_)
                @inbounds delta_v[i, particle] += delta_v_[i]
            end
       end
   end

   return system
end

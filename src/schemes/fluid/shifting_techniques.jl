abstract type AbstractShiftingTechnique end

# No shifting for a system by default
@inline shifting_technique(system) = nothing

# WARNING: Be careful if defining this function for a specific system type.
# The version for a specific system type will override this generic version.
requires_update_callback(system) = requires_update_callback(shifting_technique(system))
requires_update_callback(::Nothing) = false
requires_update_callback(::AbstractShiftingTechnique) = true

create_cache_shifting(initial_condition, ::Nothing) = (;)

function create_cache_shifting(initial_condition, ::AbstractShiftingTechnique)
    delta_v = zeros(eltype(initial_condition), ndims(initial_condition),
                    nparticles(initial_condition))

    return (; delta_v)
end

# `δv` is the correction to the particle velocity due to the shifting.
# Particles are advected with the velocity `v + δv`.
@propagate_inbounds function delta_v(system, particle)
    return delta_v(system, shifting_technique(system), particle)
end

# Zero when no shifting is used
@inline function delta_v(system, shifting, particle)
    return zero(SVector{ndims(system), eltype(system)})
end

@propagate_inbounds function delta_v(system, ::AbstractShiftingTechnique, particle)
    return extract_svector(system.cache.delta_v, system, particle)
end

function update_shifting!(system, shifting, v, u, v_ode, u_ode, semi, t)
    return system
end

# Additional term in the momentum equation due to the shifting technique
@inline function dv_shifting(shifting, system, neighbor_system,
                             particle, neighbor, v_system, v_neighbor_system,
                             m_a, m_b, rho_a, rho_b, pos_diff, distance,
                             grad_kernel, correction)
    return zero(grad_kernel)
end

@doc raw"""
    ParticleShiftingTechnique()

Particle Shifting Technique by [Sun et al. (2017)](@cite Sun2017).
Following the original paper, the callback is applied in every time step and not
in every stage of a multi-stage time integration method to reduce the computational
cost and improve the stability of the scheme.

See [Particle Shifting Technique](@ref shifting) for more information on the method.

!!! warning
    The Particle Shifting Technique needs to be disabled close to the free surface
    and therefore requires a free surface detection method. This is not yet implemented.
    **This technique cannot be used in a free surface simulation.**
"""
struct ParticleShiftingTechnique <: AbstractShiftingTechnique end

# Zero because PST is applied in a callback
@inline function delta_v(system, ::ParticleShiftingTechnique, particle)
    return zero(SVector{ndims(system), eltype(system)})
end

# This is called from the update callback
particle_shifting!(u_ode, shifting, system, semi, dt) = u_ode

function particle_shifting!(u_ode, ::ParticleShiftingTechnique, system, semi, dt)
    (; cache) = system
    (; delta_v) = cache

    u = wrap_u(u_ode, system, semi)

    # Add δr from the cache to the current coordinates
    @threaded semi for particle in eachparticle(system)
        for i in axes(delta_v, 1)
            @inbounds u[i, particle] += dt * delta_v[i, particle]
        end
    end

    return u
end

function update_shifting!(system, ::ParticleShiftingTechnique,
                          v, u, v_ode, u_ode, semi, t)
    (; cache) = system
    (; delta_v) = cache

    set_zero!(delta_v)

    # This has similar performance to `maximum(..., eachparticle(system))`,
    # but is GPU-compatible.
    v_max = maximum(x -> sqrt(dot(x, x)),
                    reinterpret(reshape, SVector{ndims(system), eltype(v)},
                                current_velocity(v, system)))

    # TODO this needs to be adapted to multi-resolution.
    # Section 3.2 explains what else needs to be changed.
    dx = particle_spacing(system, 1)
    Wdx = smoothing_kernel(system, dx, 1)
    h = smoothing_length(system, 1)

    foreach_system(semi) do neighbor_system
        u_neighbor = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor = wrap_v(v_ode, neighbor_system, semi)

        system_coords = current_coordinates(u, system)
        neighbor_coords = current_coordinates(u_neighbor, neighbor_system)

        foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords,
                               semi;
                               points=each_moving_particle(system)) do particle, neighbor,
                                                                       pos_diff, distance
            m_b = hydrodynamic_mass(neighbor_system, neighbor)
            rho_a = current_density(v, system, particle)
            rho_b = current_density(v_neighbor, neighbor_system, neighbor)

            kernel = smoothing_kernel(system, distance, particle)
            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)

            # According to p. 29 below Eq. 9
            R = 2 // 10
            n = 4

            # Eq. 7 in Sun et al. (2017).
            # According to the paper, CFL * Ma can be rewritten as Δt * v_max / h
            # (see p. 29, right above Eq. 9), but this does not work when scaling h.
            # When setting CFL * Ma = Δt * v_max / (2 * Δx), PST works as expected
            # for both small and large smoothing length factors.
            # We need to scale
            # - quadratically with the smoothing length,
            # - linearly with the particle spacing,
            # - linearly with the time step.
            # See https://github.com/trixi-framework/TrixiParticles.jl/pull/834.
            delta_v_ = -v_max * (2 * h)^2 / (2 * dx) * (1 + R * (kernel / Wdx)^n) *
                       m_b / (rho_a + rho_b) * grad_kernel

            # Write into the buffer
            for i in eachindex(delta_v_)
                @inbounds delta_v[i, particle] += delta_v_[i]
            end
        end
    end

    return system
end

"""
    TransportVelocityAdami(background_pressure::Real)

Transport Velocity Formulation (TVF) by [Adami et al. (2013)](@cite Adami2013)
to suppress pairing and tensile instability.
See [TVF](@ref transport_velocity_formulation) for more details of the method.

# Arguments
- `background_pressure`: Background pressure. Suggested is a background pressure which is
                         on the order of the reference pressure.
"""
struct TransportVelocityAdami{T <: Real} <: AbstractShiftingTechnique
    background_pressure::T
end

@inline function dv_shifting(::TransportVelocityAdami, system, neighbor_system,
                             particle, neighbor, v_system, v_neighbor_system,
                             m_a, m_b, rho_a, rho_b, pos_diff, distance,
                             grad_kernel, correction)
    v_a = current_velocity(v_system, system, particle)
    delta_v_a = delta_v(system, particle)

    v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)
    delta_v_b = delta_v(neighbor_system, neighbor)

    A_a = rho_a * v_a * delta_v_a'
    A_b = rho_b * v_b * delta_v_b'

    # The following term depends on the pressure acceleration formulation.
    # See the large comment below. In the original paper (Adami et al., 2013), this is
    #   (V_a^2 + V_b^2) / m_a * ((A_a + A_b) / 2) * ∇W_ab.
    # With the most common pressure acceleration formulation, this is
    #   m_b * (A_a + A_b) / (ρ_a * ρ_b) * ∇W_ab.
    # In order to obtain this, we pass `p_a = A_a` and `p_b = A_b` to the
    # `pressure_acceleration` function.
    return pressure_acceleration(system, neighbor_system, particle, neighbor,
                                 m_a, m_b, A_a, A_b, rho_a, rho_b, pos_diff,
                                 distance, grad_kernel, correction)
end

function update_shifting!(system, shifting::TransportVelocityAdami, v, u, v_ode,
                          u_ode, semi, t)
    (; cache, correction) = system
    (; delta_v) = cache
    (; background_pressure) = shifting

    sound_speed = system_sound_speed(system)

    set_zero!(delta_v)

    foreach_system(semi) do neighbor_system
        v_neighbor = wrap_v(v_ode, neighbor_system, semi)
        u_neighbor = wrap_u(u_ode, neighbor_system, semi)

        system_coords = current_coordinates(u, system)
        neighbor_coords = current_coordinates(u_neighbor, neighbor_system)

        foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords,
                               semi;
                               points=each_moving_particle(system)) do particle, neighbor,
                                                                       pos_diff, distance
            m_a = @inbounds hydrodynamic_mass(system, particle)
            m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)

            rho_a = @inbounds current_density(v, system, particle)
            rho_b = @inbounds current_density(v_neighbor, neighbor_system, neighbor)

            h = smoothing_length(system, particle)

            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)

            # In the original paper (Adami et al., 2013), the transport velocity is applied
            # as follows:
            #   v_{1/2} = v_0 + Δt/2 * a,
            # where a is the regular SPH acceleration term (pressure, viscosity, etc.).
            #   r_1 = r_0 + Δt * (v_{1/2},
            # where ̃v_{1/2} = v_{1/2} + Δt/2 * p_0 / m_a * \sum_b[ (V_a^2 + V_b^2) * ∇W_ab ]
            # is the transport velocity.
            # We call δv_{1/2} = ̃v_{1/2} - v_{1/2} the shifting velocity.
            # We will call δv_{1/2} the shifting velocity, which is given by
            #   δv = -Δt/2 * p_0 / m_a * \sum_b[ (V_a^2 + V_b^2) * ∇W_ab ],
            # where p_0 is the background pressure, V_a = m_a / ρ_a, V_b = m_b / ρ_b.
            # This term depends on the pressure acceleration formulation.
            # In Zhang et al. (2017), the pressure acceleration term
            #   m_b * (p_a / ρ_a^2 + p_b / ρ_b^2) * ∇W_ab
            # is used. They consequently changed the shifting velocity to
            #   δv = -Δt/2 * p_0 * \sum_b[ m_b * (1 / ρ_a^2 + 1 / ρ_b^2) * ∇W_ab ].
            # We therefore use the function `pressure_acceleration` to compute the
            # shifting velocity according to the used pressure acceleration formulation.
            # In most cases, this will be
            #   δv = -Δt/2 * p_0 * \sum_b[ m_b * (1 + 1) / (ρ_a * ρ_b) * ∇W_ab ].
            #
            # In these papers, the shifting velocity is scaled by the time step Δt.
            # We generally want the spatial discretization to be independent of the time step.
            # Scaling the shifting velocity by the time step would lead to less shifting
            # when very small time steps are used for testing/debugging purposes.
            # This is especially problematic in TrixiParticles.jl, as the time step can vary
            # significantly between different time integration methods (low vs high order).
            # In order to eliminate the time step from the shifting velocity, we apply the
            # CFL condition used in Adami et al. (2013):
            #   Δt <= 0.25 * h / c,
            # where h is the smoothing length and c is the sound speed.
            # Applying this equation as equality yields the shifting velocity
            #   δv = -p_0 / 8 * h / c * \sum_b[ m_b * (1 + 1) / (ρ_a * ρ_b) * ∇W_ab ].
            # The last part is achieved by passing `p_a = 1` and `p_b = 1` to the
            # `pressure_acceleration` function.
            delta_v_ = background_pressure / 8 * h / sound_speed *
                       pressure_acceleration(system, neighbor_system, particle, neighbor,
                                             m_a, m_b, 1, 1, rho_a, rho_b, pos_diff,
                                             distance, grad_kernel, correction)

            # Write into the buffer
            for i in eachindex(delta_v_)
                @inbounds delta_v[i, particle] += delta_v_[i]
            end
        end
    end

    return system
end

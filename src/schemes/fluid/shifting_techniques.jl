abstract type AbstractShiftingTechnique end

# No shifting for a system by default
@inline shifting_technique(system) = nothing

# WARNING: Be careful if defining this function for a specific system type.
# The version for a specific system type will override this generic version.
requires_update_callback(system) = requires_update_callback(shifting_technique(system))
requires_update_callback(::Nothing) = false
requires_update_callback(::AbstractShiftingTechnique) = true

# This is called from the `UpdateCallback`
particle_shifting_from_callback!(u_ode, shifting, system, v_ode, semi, dt) = u_ode

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

function update_shifting!(system, shifting, v, u, v_ode, u_ode, semi)
    return system
end

# Additional term in the momentum equation due to the shifting technique
@inline function dv_shifting(shifting, system, neighbor_system,
                             v_system, v_neighbor_system, particle, neighbor,
                             m_a, m_b, rho_a, rho_b, pos_diff, distance,
                             grad_kernel, correction)
    return zero(grad_kernel)
end

# Additional term(s) in the continuity equation due to the shifting technique
function continuity_equation_shifting!(dv, shifting,
                                       particle_system, neighbor_system,
                                       particle, neighbor, grad_kernel, rho_a, rho_b, m_b)
    return dv
end

@doc raw"""
    ParticleShiftingTechnique(; integrate_shifting_velocity=true,
                              update_everystage=false,
                              modify_continuity_equation=true,
                              second_continuity_equation_term=true,
                              modify_momentum_equation=true,
                              v_max_factor=1, sound_speed_factor=0)

Particle Shifting Technique by [Sun et al. (2017)](@cite Sun2017)
and [Sun et al. (2019)](@cite Sun2019).
The keyword arguments allow to choose between the original methods from the two papers
and variants in between.

The default values of the keyword arguments provide the version of shifting
that we recommend based on our experiments.
The default values are subject to change in future releases.

See [Particle Shifting Technique](@ref shifting) for more information on the method.

We provide the following convenience constructors for common variants of the method:
- [`ParticleShiftingTechniqueSun2017()`](@ref):
    Particle Shifting Technique by [Sun et al. (2017)](@cite Sun2017).
    Shifting is applied as a position correction in a callback after each time step.
    No additional terms are added to the momentum or continuity equations.
- [`ConsistentShiftingSun2019()`](@ref):
    Consistent Particle Shifting Technique by [Sun et al. (2019)](@cite Sun2019).
    Shifting is applied with a shifting velocity in each stage of the time integration.
    Additional terms are added to the momentum and continuity equations, most importantly
    to guarantee conservation of volume in closed systems, which is not the case for the
    original method by [Sun et al. (2017)](@cite Sun2017).

# Keywords
- `integrate_shifting_velocity`: If `true`, the shifting is applied in each stage of the
                                time integration method as a shifting velocity that is
                                added to the physical velocity in the time integration.
                                If `false`, the shifting is applied as a position correction
                                in a callback after each time step.
- `update_everystage`:          If `true`, the shifting velocity is updated in every stage
                                of a multi-stage time integration method.
                                This requires `integrate_shifting_velocity=true`.
                                If `false`, the shifting velocity is only updated once
                                per time step in a callback, and the same shifting velocity
                                is used for all stages.
                                `update_everystage=false` reduces the computational cost,
                                but may reduce the stability of the scheme and require
                                a smaller time step.
- `modify_continuity_equation`: If `true`, the continuity equation is modified to be based
                                on the transport velocity instead of the physical velocity.
                                This guarantees conservation of volume in closed systems,
                                but is unstable at solid wall boundaries, according to our
                                experiments.
                                This requires `integrate_shifting_velocity=true`.
- `second_continuity_equation_term`: If `true`, a second term
                                by [Sun et al. (2019)](@cite Sun2019) is added to the
                                continuity equation to solve the stability problems with
                                the modified continuity equation at solid wall boundaries.
                                This requires `modify_continuity_equation=true`.
- `modify_momentum_equation`:   If `true`, an additional term
                                by [Sun et al. (2019)](@cite Sun2019) is added
                                to the momentum equation.
                                This requires `integrate_shifting_velocity=true`.
- `v_max_factor`:               Factor to scale the expected maximum velocity used in the
                                shifting velocity. The maximum expected velocity is computed as
                                `v_max_factor * max(|v|)`, where `v` is the physical velocity.
                                As opposed to `sound_speed_factor`, the computed expected
                                maximum velocity depends on the current flow field
                                and can change over time.
                                Only one of `v_max_factor` and `sound_speed_factor`
                                can be non-zero.
- `sound_speed_factor`:         Factor to compute the maximum expected velocity used in the
                                shifting velocity from the speed of sound.
                                The maximum expected velocity is computed as
                                `sound_speed_factor * c`, where `c` is the speed of sound.
                                Only one of `v_max_factor` and `sound_speed_factor`
                                can be non-zero.

!!! warning
    The Particle Shifting Technique needs to be disabled close to the free surface
    and therefore requires a free surface detection method. This is not yet implemented.
    **This technique cannot be used in a free surface simulation.**
"""
struct ParticleShiftingTechnique{integrate_shifting_velocity,
                                 update_everystage,
                                 modify_continuity_equation,
                                 second_continuity_equation_term,
                                 modify_momentum_equation,
                                 compute_v_max,
                                 ELTYPE} <: AbstractShiftingTechnique
    v_factor::ELTYPE

    function ParticleShiftingTechnique(; integrate_shifting_velocity=true,
                                       update_everystage=false,
                                       modify_continuity_equation=true,
                                       second_continuity_equation_term=true,
                                       modify_momentum_equation=true,
                                       v_max_factor=1, sound_speed_factor=0)
        if !integrate_shifting_velocity && update_everystage
            throw(ArgumentError("ParticleShiftingTechnique: " *
                                "integrate_shifting_velocity=false requires " *
                                "update_everystage=false"))
        end

        if !integrate_shifting_velocity && modify_continuity_equation
            throw(ArgumentError("ParticleShiftingTechnique: " *
                                "modify_continuity_equation=true requires " *
                                "integrate_shifting_velocity=true"))
        end

        if !modify_continuity_equation && second_continuity_equation_term
            throw(ArgumentError("ParticleShiftingTechnique: " *
                                "second_continuity_equation_term=true requires " *
                                "modify_continuity_equation=true"))
        end

        if !integrate_shifting_velocity && modify_momentum_equation
            throw(ArgumentError("ParticleShiftingTechnique: " *
                                "modify_momentum_equation=true requires " *
                                "integrate_shifting_velocity=true"))
        end

        if v_max_factor > 0 && sound_speed_factor > 0
            throw(ArgumentError("ParticleShiftingTechnique: " *
                                "Only one of v_max_factor and sound_speed_factor " *
                                "can be non-zero"))
        end

        if v_max_factor <= 0 && sound_speed_factor <= 0
            throw(ArgumentError("ParticleShiftingTechnique: " *
                                "One of v_max_factor and sound_speed_factor " *
                                "must be positive"))
        end

        v_factor = max(v_max_factor, sound_speed_factor)
        compute_v_max = v_max_factor > 0

        new{integrate_shifting_velocity,
            update_everystage,
            modify_continuity_equation,
            second_continuity_equation_term,
            modify_momentum_equation,
            compute_v_max, typeof(v_factor)}(v_factor)
    end
end

"""
    ParticleShiftingTechniqueSun2017(; kwargs...)

Particle Shifting Technique by [Sun et al. (2017)](@cite Sun2017).
Following the original paper, the callback is applied in every time step and not
in every stage of a multi-stage time integration method to reduce the computational cost.

This is a convenience constructor for:
```jldoctest; output = false
ParticleShiftingTechnique(integrate_shifting_velocity=false,
                          update_everystage=false,
                          modify_continuity_equation=false,
                          second_continuity_equation_term=false,
                          modify_momentum_equation=false,
                          v_max_factor=1, sound_speed_factor=0)

# output
ParticleShiftingTechnique{false, false, false, false, false, true, Int64}(1)
```

See [ParticleShiftingTechnique](@ref ParticleShiftingTechnique) for all available options.

# Keywords
- `kwargs...`: All keywords are passed to the main constructor.

# Examples
```jldoctest; output = false
shifting_technique = ParticleShiftingTechniqueSun2017()

# output
ParticleShiftingTechnique{false, false, false, false, false, true, Int64}(1)
```

!!! warning
    The Particle Shifting Technique needs to be disabled close to the free surface
    and therefore requires a free surface detection method. This is not yet implemented.
    **This technique cannot be used in a free surface simulation.**
"""
function ParticleShiftingTechniqueSun2017(; kwargs...)
    return ParticleShiftingTechnique(; integrate_shifting_velocity=false,
                                     update_everystage=false,
                                     modify_continuity_equation=false,
                                     second_continuity_equation_term=false,
                                     modify_momentum_equation=false,
                                     v_max_factor=1, sound_speed_factor=0,
                                     kwargs...)
end

"""
    ConsistentShiftingSun2019(; sound_speed_factor=0.1, kwargs...)

Consistent Particle Shifting Technique by [Sun et al. (2019)](@cite Sun2019).

This is a convenience constructor for:
```jldoctest; output = false
ParticleShiftingTechnique(integrate_shifting_velocity=true,
                          update_everystage=true,
                          modify_continuity_equation=true,
                          second_continuity_equation_term=true,
                          modify_momentum_equation=true,
                          v_max_factor=0, sound_speed_factor=0.1)

# output
ParticleShiftingTechnique{true, true, true, true, true, false, Float64}(0.1)
```

See [ParticleShiftingTechnique](@ref ParticleShiftingTechnique) for all available options.

# Keywords
- `sound_speed_factor`: Factor to compute the maximum expected velocity used in the
                        shifting velocity from the speed of sound.
                        The maximum expected velocity is computed as
                        `sound_speed_factor * c`, where `c` is the speed of sound.
                        Since the speed of sound is usually chosen as 10 times the maximum
                        expected velocity in weakly compressible SPH, a value of 0.1
                        corresponds to the maximum expected velocity.
- `kwargs...`: All keywords are passed to the main constructor.

# Examples
```jldoctest; output = false
shifting_technique = ConsistentShiftingSun2019()

# output
ParticleShiftingTechnique{true, true, true, true, true, false, Float64}(0.1)
```

!!! warning
    The Particle Shifting Technique needs to be disabled close to the free surface
    and therefore requires a free surface detection method. This is not yet implemented.
    **This technique cannot be used in a free surface simulation.**
"""
function ConsistentShiftingSun2019(; kwargs...)
    return ParticleShiftingTechnique(; integrate_shifting_velocity=true,
                                     update_everystage=true,
                                     modify_continuity_equation=true,
                                     second_continuity_equation_term=true,
                                     modify_momentum_equation=true,
                                     v_max_factor=0, sound_speed_factor=0.1,
                                     kwargs...)
end

# `ParticleShiftingTechnique{false}` means `integrate_shifting_velocity=false`.
# Zero if PST is applied in a callback as a position correction
# and not with a shifting velocity in the time integration stages
# (which would be `integrate_shifting_velocity=false`).
@inline function delta_v(system, ::ParticleShiftingTechnique{false}, particle)
    return zero(SVector{ndims(system), eltype(system)})
end

# `ParticleShiftingTechnique{<:Any, <:Any, <:Any, <:Any, true}` means
# `modify_momentum_equation=true`.
@propagate_inbounds function dv_shifting(::ParticleShiftingTechnique{<:Any, <:Any,
                                                                     <:Any, <:Any, true},
                                         system, neighbor_system,
                                         v_system, v_neighbor_system,
                                         particle, neighbor, m_a, m_b, rho_a, rho_b,
                                         pos_diff, distance, grad_kernel, correction)
    delta_v_a = delta_v(system, particle)
    delta_v_b = delta_v(neighbor_system, neighbor)

    v_a = current_velocity(v_system, system, particle)
    v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)

    tensor_product = v_a * delta_v_a' + v_b * delta_v_b'
    return m_b / rho_b *
           (tensor_product * grad_kernel + v_a * dot(delta_v_a - delta_v_b, grad_kernel))
end

# `ParticleShiftingTechnique{<:Any, <:Any, true}` means `modify_continuity_equation=true`
function continuity_equation_shifting!(dv,
                                       shifting::ParticleShiftingTechnique{<:Any, <:Any,
                                                                           true},
                                       system, neighbor_system,
                                       particle, neighbor, grad_kernel, rho_a, rho_b, m_b)
    delta_v_diff = delta_v(system, particle) -
                   delta_v(neighbor_system, neighbor)

    dv[end, particle] += rho_a / rho_b * m_b * dot(delta_v_diff, grad_kernel)

    second_continuity_equation_term!(dv, shifting,
                                     system, neighbor_system,
                                     particle, neighbor, grad_kernel, rho_a, rho_b, m_b)

    return dv
end

# `ParticleShiftingTechnique{<:Any, <:Any, <:Any, true}` means
# `second_continuity_equation_term=true`.
@inline function second_continuity_equation_term!(dv,
                                                  ::ParticleShiftingTechnique{<:Any, <:Any,
                                                                              <:Any, true},
                                                  system, neighbor_system,
                                                  particle, neighbor, grad_kernel,
                                                  rho_a, rho_b, m_b)
    rho_v = rho_a * delta_v(system, particle) + rho_b * delta_v(neighbor_system, neighbor)

    dv[end, particle] += m_b / rho_b * dot(rho_v, grad_kernel)

    return dv
end

@inline function second_continuity_equation_term!(dv, shifting,
                                                  system, neighbor_system,
                                                  particle, neighbor, grad_kernel,
                                                  rho_a, rho_b, m_b)
    return dv
end

# `ParticleShiftingTechnique{<:Any, true}` means `update_everystage=true`
function update_shifting!(system, shifting::ParticleShiftingTechnique{<:Any, true},
                          v, u, v_ode, u_ode, semi)
    update_shifting_inner!(system, shifting, v, u, v_ode, u_ode, semi)
end

# `ParticleShiftingTechnique{<:Any, false}` means `update_everystage=false`
function update_shifting_from_callback!(system,
                                        shifting::ParticleShiftingTechnique{<:Any, false},
                                        v_ode, u_ode, semi)
    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    update_shifting_inner!(system, shifting, v, u, v_ode, u_ode, semi)
end

# `ParticleShiftingTechnique{<:Any, <:Any, <:Any, <:Any, <:Any, true}`
# means `compute_v_max=true`
function v_max(shifting::ParticleShiftingTechnique{<:Any, <:Any, <:Any, <:Any, <:Any, true},
               v, system)
    # This has similar performance to `maximum(..., eachparticle(system))`,
    # but is GPU-compatible.
    v_max = maximum(x -> sqrt(dot(x, x)),
                    reinterpret(reshape, SVector{ndims(system), eltype(v)},
                                current_velocity(v, system)))
    return shifting.v_factor * v_max
end

# `ParticleShiftingTechnique{<:Any, <:Any, <:Any, <:Any, <:Any, false}`
# means `compute_v_max=false`
function v_max(shifting::ParticleShiftingTechnique{<:Any, <:Any, <:Any, <:Any, <:Any,
                                                   false},
               v, system)
    sound_speed = system_sound_speed(system)

    return shifting.v_factor * sound_speed
end

function update_shifting_inner!(system, shifting::ParticleShiftingTechnique,
                                v, u, v_ode, u_ode, semi)
    (; cache) = system
    (; delta_v) = cache

    set_zero!(delta_v)

    v_max_ = v_max(shifting, v, system)

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
            # (see p. 29, right above Eq. 9), but this does not yield the same amount
            # of shifting when scaling h.
            # When setting CFL * Ma = Δt * v_max / (2 * Δx), PST works as expected
            # for both small and large smoothing length factors.
            # We need to scale
            # - quadratically with the smoothing length,
            # - linearly with the particle spacing,
            # - linearly with the time step.
            # See https://github.com/trixi-framework/TrixiParticles.jl/pull/834.
            delta_v_ = -v_max_ * (2 * h)^2 / (2 * dx) * (1 + R * (kernel / Wdx)^n) *
                       m_b / (rho_a + rho_b) * grad_kernel

            # Write into the buffer
            for i in eachindex(delta_v_)
                @inbounds delta_v[i, particle] += delta_v_[i]
            end
        end
    end

    return system
end

# `ParticleShiftingTechnique{<:Any, false}` means `update_everystage=false`.
# Only update shifting from callback if `update_everystage=false`.
# Only apply shifting from callback if PST is to be applied in a callback
# (`integrate_shifting_velocity=false`), but this also requires `update_everystage=false`.
function particle_shifting_from_callback!(u_ode,
                                          shifting::ParticleShiftingTechnique{<:Any, false},
                                          system, v_ode, semi, dt)
    @trixi_timeit timer() "particle shifting" begin
        # Update the shifting velocity
        update_shifting_from_callback!(system, shifting, v_ode, u_ode, semi)

        # Update the particle positions with the shifting velocity
        apply_particle_shifting!(u_ode, shifting, system, semi, dt)
    end
end

# `ParticleShiftingTechnique{false}` means `integrate_shifting_velocity=false`.
# Only apply shifting from callback if PST is to be applied in a callback
# and not with a shifting velocity in the time integration stages
# (which would be `integrate_shifting_velocity=false`).
function apply_particle_shifting!(u_ode, ::ParticleShiftingTechnique{false},
                                  system, semi, dt)
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

function apply_particle_shifting!(u_ode, ::ParticleShiftingTechnique{true},
                                  system, semi, dt)
    return u_ode
end

"""
    TransportVelocityAdami(; background_pressure::Real, modify_continuity_equation=false)

Transport Velocity Formulation (TVF) by [Adami et al. (2013)](@cite Adami2013)
to suppress pairing and tensile instability.
See [TVF](@ref transport_velocity_formulation) for more details of the method.

# Keywords
- `background_pressure`: Background pressure. Suggested is a background pressure which is
                         on the order of the reference pressure.
- `modify_continuity_equation`: If `true`, the continuity equation is modified to be based
                                on the transport velocity instead of the physical velocity.
                                This guarantees conservation of volume in closed systems,
                                but is unstable at solid wall boundaries, according to our
                                experiments.

!!! warning
    The Transport Velocity Formulation needs to be disabled close to the free surface
    and therefore requires a free surface detection method. This is not yet implemented.
    **This technique cannot be used in a free surface simulation.**
"""
struct TransportVelocityAdami{modify_continuity_equation, T <: Real} <:
       AbstractShiftingTechnique
    background_pressure::T

    function TransportVelocityAdami(; background_pressure, modify_continuity_equation=false)
        new{modify_continuity_equation, typeof(background_pressure)}(background_pressure)
    end
end

@propagate_inbounds function dv_shifting(::TransportVelocityAdami, system, neighbor_system,
                                         v_system, v_neighbor_system, particle, neighbor,
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

function continuity_equation_shifting!(dv, shifting::TransportVelocityAdami{true},
                                       particle_system, neighbor_system,
                                       particle, neighbor, grad_kernel, rho_a, rho_b, m_b)
    delta_v_diff = delta_v(particle_system, particle) -
                   delta_v(neighbor_system, neighbor)

    dv[end, particle] += rho_a / rho_b * m_b * dot(delta_v_diff, grad_kernel)

    return dv
end

function update_shifting!(system, shifting::TransportVelocityAdami, v, u, v_ode,
                          u_ode, semi)
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

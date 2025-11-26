abstract type AbstractShiftingTechnique end

# No shifting for a system by default
@inline shifting_technique(system) = nothing

# WARNING: Be careful if defining this function for a specific system type.
# The version for a specific system type will override this generic version.
requires_update_callback(system) = requires_update_callback(shifting_technique(system))
requires_update_callback(::Nothing) = false
requires_update_callback(::AbstractShiftingTechnique) = false

# This is called from the `UpdateCallback`
particle_shifting_from_callback!(u_ode, shifting, system, v_ode, semi, integrator) = u_ode

create_cache_shifting(initial_condition, ::Nothing) = (;)

function create_cache_shifting(initial_condition, ::AbstractShiftingTechnique)
    delta_v = zeros(eltype(initial_condition), ndims(initial_condition),
                    nparticles(initial_condition))

    nct = zeros(eltype(initial_condition), nparticles(initial_condition))

    return (; delta_v, nct)
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
@inline function continuity_equation_shifting_term(shifting, particle_system,
                                                   neighbor_system,
                                                   particle, neighbor, rho_a, rho_b)
    return zero(SVector{ndims(particle_system), eltype(particle_system)})
end

@doc raw"""
    ParticleShiftingTechnique(; integrate_shifting_velocity=true,
                              update_everystage=false,
                              modify_continuity_equation=true,
                              second_continuity_equation_term=ContinuityEquationTermSun2019(),
                              momentum_equation_term=MomentumEquationTermSun2019(),
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
- `second_continuity_equation_term`: Additional term to be added to the
                                continuity equation to solve the stability problems with
                                the modified continuity equation at solid wall boundaries.
                                This requires `modify_continuity_equation=true`.
                                See [`ContinuityEquationTermSun2019`](@ref).
- `momentum_equation_term`:     Additional term to be added to the momentum equation
                                to account for the shifting velocity.
                                This requires `integrate_shifting_velocity=true`.
                                See [`MomentumEquationTermSun2019`](@ref).
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
                                 compute_v_max,
                                 ELTYPE, S, M} <: AbstractShiftingTechnique
    v_factor                        :: ELTYPE
    second_continuity_equation_term :: S
    momentum_equation_term          :: M

    function ParticleShiftingTechnique(; integrate_shifting_velocity=true,
                                       update_everystage=false,
                                       modify_continuity_equation=true,
                                       second_continuity_equation_term=ContinuityEquationTermSun2019(),
                                       momentum_equation_term=MomentumEquationTermSun2019(),
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

        if !modify_continuity_equation && !isnothing(second_continuity_equation_term)
            throw(ArgumentError("ParticleShiftingTechnique: " *
                                "a second_continuity_equation_term requires " *
                                "modify_continuity_equation=true"))
        end

        if !integrate_shifting_velocity && !isnothing(momentum_equation_term)
            throw(ArgumentError("ParticleShiftingTechnique: " *
                                "a momentum_equation_term requires " *
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
            compute_v_max, typeof(v_factor),
            typeof(second_continuity_equation_term),
            typeof(momentum_equation_term)}(v_factor,
                                            second_continuity_equation_term,
                                            momentum_equation_term)
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
                          second_continuity_equation_term=nothing,
                          momentum_equation_term=nothing,
                          v_max_factor=1, sound_speed_factor=0)

# output
ParticleShiftingTechnique{false, false, false, true, Int64, Nothing, Nothing}(1, nothing, nothing)
```

See [ParticleShiftingTechnique](@ref ParticleShiftingTechnique) for all available options.

# Keywords
- `kwargs...`: All keywords are passed to the main constructor.

# Examples
```jldoctest; output = false
shifting_technique = ParticleShiftingTechniqueSun2017()

# output
ParticleShiftingTechnique{false, false, false, true, Int64, Nothing, Nothing}(1, nothing, nothing)
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
                                     second_continuity_equation_term=nothing,
                                     momentum_equation_term=nothing,
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
                          second_continuity_equation_term=ContinuityEquationTermSun2019(),
                          momentum_equation_term=MomentumEquationTermSun2019(),
                          v_max_factor=0, sound_speed_factor=0.1)

# output
ParticleShiftingTechnique{true, true, true, false, Float64, ContinuityEquationTermSun2019, MomentumEquationTermSun2019}(0.1, ContinuityEquationTermSun2019(), MomentumEquationTermSun2019())
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
ParticleShiftingTechnique{true, true, true, false, Float64, ContinuityEquationTermSun2019, MomentumEquationTermSun2019}(0.1, ContinuityEquationTermSun2019(), MomentumEquationTermSun2019())
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
                                     second_continuity_equation_term=ContinuityEquationTermSun2019(),
                                     momentum_equation_term=MomentumEquationTermSun2019(),
                                     v_max_factor=0, sound_speed_factor=0.1,
                                     kwargs...)
end

# `ParticleShiftingTechnique{<:Any, false}` means `update_everystage=false`,
# so the `UpdateCallback` is required to update the shifting velocity.
requires_update_callback(::ParticleShiftingTechnique{<:Any, false}) = true

# `ParticleShiftingTechnique{false}` means `integrate_shifting_velocity=false`.
# Zero if PST is applied in a callback as a position correction
# and not with a shifting velocity in the time integration stages
# (which would be `integrate_shifting_velocity=false`).
@inline function delta_v(system, ::ParticleShiftingTechnique{false}, particle)
    return zero(SVector{ndims(system), eltype(system)})
end

"""
    MomentumEquationTermSun2019()

A term by [Sun et al. (2019)](@cite Sun2019) to be added to the momentum equation.

See [`ParticleShiftingTechnique`](@ref).
"""
struct MomentumEquationTermSun2019 end

# Additional term in the momentum equation due to the shifting technique
@propagate_inbounds function dv_shifting(shifting::ParticleShiftingTechnique, system,
                                         neighbor_system,
                                         v_system, v_neighbor_system, particle, neighbor,
                                         m_a, m_b, rho_a, rho_b, pos_diff, distance,
                                         grad_kernel, correction)
    return dv_shifting(shifting.momentum_equation_term, system, neighbor_system,
                       v_system, v_neighbor_system, particle, neighbor,
                       m_a, m_b, rho_a, rho_b, pos_diff, distance,
                       grad_kernel, correction)
end

@propagate_inbounds function dv_shifting(::MomentumEquationTermSun2019,
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
@propagate_inbounds function continuity_equation_shifting_term(shifting::ParticleShiftingTechnique{<:Any,
                                                                                                   <:Any,
                                                                                                   true},
                                                               system, neighbor_system,
                                                               particle, neighbor,
                                                               rho_a, rho_b)
    delta_v_a = delta_v(system, particle)
    delta_v_b = delta_v(neighbor_system, neighbor)
    delta_v_diff = delta_v_a - delta_v_b

    second_term = second_continuity_equation_term(shifting.second_continuity_equation_term,
                                                  delta_v_a, delta_v_b, rho_a, rho_b)
    return delta_v_diff + second_term
end

"""
    ContinuityEquationTermSun2019()

A second term by [Sun et al. (2019)](@cite Sun2019) to be added to the continuity equation
to solve the stability problems with the modified continuity equation at solid wall boundaries.

See [`ParticleShiftingTechnique`](@ref).
"""
struct ContinuityEquationTermSun2019 end

@propagate_inbounds function second_continuity_equation_term(::ContinuityEquationTermSun2019,
                                                             delta_v_a, delta_v_b,
                                                             rho_a, rho_b)
    return delta_v_a + rho_b / rho_a * delta_v_b
end

@inline function second_continuity_equation_term(second_continuity_equation_term,
                                                 delta_v_a, delta_v_b, rho_a, rho_b)
    return zero(delta_v_a)
end

# `ParticleShiftingTechnique{<:Any, true}` means `update_everystage=true`
function update_shifting!(system, shifting::ParticleShiftingTechnique{<:Any, true},
                          v, u, v_ode, u_ode, semi)
    @trixi_timeit timer() "update shifting" begin
        update_shifting_inner!(system, shifting, v, u, v_ode, u_ode, semi)
    end
end

# `ParticleShiftingTechnique{<:Any, false}` means `update_everystage=false`
function update_shifting_from_callback!(system,
                                        shifting::ParticleShiftingTechnique{<:Any, false},
                                        v_ode, u_ode, semi)
    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    update_shifting_inner!(system, shifting, v, u, v_ode, u_ode, semi)
end

# `ParticleShiftingTechnique{<:Any, <:Any, <:Any, true}`
# means `compute_v_max=true`
function v_max(shifting::ParticleShiftingTechnique{<:Any, <:Any, <:Any, true},
               v, system)
    # This has similar performance as `maximum(..., eachparticle(system))`,
    # but is GPU-compatible.
    v_max2 = maximum(x -> dot(x, x),
                     reinterpret(reshape, SVector{ndims(system), eltype(v)},
                                 current_velocity(v, system)))
    v_max = sqrt(v_max2)

    return shifting.v_factor * v_max
end

# `ParticleShiftingTechnique{<:Any, <:Any, <:Any, false}`
# means `compute_v_max=false`
function v_max(shifting::ParticleShiftingTechnique{<:Any, <:Any, <:Any, false},
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
                               points=each_integrated_particle(system)) do particle,
                                                                           neighbor,
                                                                           pos_diff,
                                                                           distance
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

    modify_shifting_at_free_surfaces!(system, u, semi, u_ode)

    return system
end

# `ParticleShiftingTechnique{<:Any, false}` means `update_everystage=false`.
# Only update shifting from callback if `update_everystage=false`.
# Only apply shifting from callback if PST is to be applied in a callback
# (`integrate_shifting_velocity=false`), but this also requires `update_everystage=false`.
function particle_shifting_from_callback!(u_ode,
                                          shifting::ParticleShiftingTechnique{<:Any, false},
                                          system, v_ode, semi, integrator)
    # Update the shifting velocity
    update_shifting_from_callback!(system, shifting, v_ode, u_ode, semi)

    @trixi_timeit timer() "apply particle shifting" begin
        # Update the particle positions with the shifting velocity
        apply_particle_shifting!(u_ode, shifting, system, semi, integrator.dt)
    end

    # Tell OrdinaryDiffEq that `integrator.u` has been modified
    u_modified!(integrator, true)

    return u_ode
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

# The function above misuses the pressure acceleration function by passing a Matrix as `p_a`.
# This doesn't work with `tensile_instability_control`, so we disable TIC in this case.
@inline function tensile_instability_control(m_a, m_b, rho_a, rho_b, p_a::SMatrix, p_b, W_a)
    return pressure_acceleration_continuity_density(m_a, m_b, rho_a, rho_b, p_a, p_b, W_a)
end

@propagate_inbounds function continuity_equation_shifting_term(::TransportVelocityAdami{true},
                                                               particle_system,
                                                               neighbor_system,
                                                               particle, neighbor,
                                                               rho_a, rho_b)
    delta_v_diff = delta_v(particle_system, particle) -
                   delta_v(neighbor_system, neighbor)

    return delta_v_diff
end

function update_shifting!(system, shifting::TransportVelocityAdami, v, u, v_ode,
                          u_ode, semi)
    (; delta_v) = system.cache
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
                               points=each_integrated_particle(system)) do particle,
                                                                           neighbor,
                                                                           pos_diff,
                                                                           distance
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
                                             distance, grad_kernel,
                                             system_correction(system))

            # Write into the buffer
            for i in eachindex(delta_v_)
                @inbounds delta_v[i, particle] += delta_v_[i]
            end
        end
    end

    modify_shifting_at_free_surfaces!(system, u, semi, u_ode)

    return system
end

# TODO: Implement free surface detection to disable shifting close to free surfaces
# function modify_shifting_at_free_surfaces!(system, u, semi, u_ode)
#     T   = eltype(system.cache.delta_v)
#     dim = ndims(system)
#     np  = size(system.cache.delta_v, 2)
#     epsT = eps(T)

#     # ------------------------------------------------------------
#     # PASS 1: free-surface indicators & wall stats
#     #   - psi_nog:   Σ W_ab (EXCLUDING boundary systems)  → limiter/support
#     #   - gsum_wg:   Σ ∇W_ab (INCLUDING ghosts)           → FS normals
#     #   - psi_wall:  Σ W_ab (boundary only)               → wall ratio/damping
#     #   - S (2x2):   Σ (∇W)(∇W)^T (boundary only)         → corner detector
#     # ------------------------------------------------------------
#     coordsA  = current_coordinates(u, system)
#     psi_nog  = zeros(T, np)
#     gsum_wg  = zeros(T, dim, np)
#     psi_wall = zeros(T, np)
#     S11 = zeros(T, np); S12 = zeros(T, np); S22 = zeros(T, np)  # used in 2D

#     foreach_system(semi) do sysB
#         # if u_ode not passed, we can only see same-system neighbors
#         if (u_ode === nothing) && (sysB !== system)
#             return
#         end
#         uB      = (u_ode === nothing) ? u : wrap_u(u_ode, sysB, semi)
#         coordsB = current_coordinates(uB, sysB)

#         foreach_point_neighbor(system, sysB, coordsA, coordsB, semi;
#                                points=each_integrated_particle(system)) do a,b,pos_diff,distance
#             Wab  = smoothing_kernel(system, distance, a)
#             gWab = smoothing_kernel_grad(system, pos_diff, distance, a)

#             # WITH ghosts → robust normals near walls
#             @inbounds for i in 1:dim
#                 gsum_wg[i, a] += gWab[i]
#             end

#             if sysB isa AbstractBoundarySystem
#                 @inbounds psi_wall[a] += Wab
#                 if dim >= 2
#                     @inbounds begin
#                         S11[a] += gWab[1]*gWab[1]
#                         S12[a] += gWab[1]*gWab[2]
#                         S22[a] += gWab[2]*gWab[2]
#                     end
#                 end
#             else
#                 @inbounds psi_nog[a] += Wab   # support WITHOUT ghosts
#             end
#         end
#     end

#     smax_nog  = maximum(psi_nog)
#     h         = smoothing_length(system, 1)

#     # Tunables (more permissive than before)
#     λ_min     = T(0.50)   # relaxed cutoff (was 0.55)
#     α_mask    = T(0.85)   # FS detection threshold (unchanged)
#     α_limit   = T(0.80)   # NEW: softer limiter denominator → more δu
#     τ_grad    = T(0.30)
#     s_min     = T(0.15)   # NEW: minimum limiter floor so FS keeps some δu

#     # FS mask: detect using support (no-ghosts) and gradient magnitude (with-ghosts)
#     FS = falses(np)
#     @inbounds for a in 1:np
#         g2 = zero(T)
#         for i in 1:dim
#             g2 += gsum_wg[i, a]^2
#         end
#         gnorm = sqrt(g2)
#         FS[a] = (psi_nog[a] < α_mask * smax_nog) || (h * gnorm > τ_grad)
#     end

#     # Unit normals from WITH-ghosts gradient sum (only where FS=true)
#     n_hat = zeros(T, dim, np)
#     @inbounds for a in 1:np
#         if FS[a]
#             g2 = zero(T)
#             for i in 1:dim
#                 g2 += gsum_wg[i, a]^2
#             end
#             gnorm = sqrt(g2)
#             if gnorm > T(1e2)*epsT
#                 for i in 1:dim
#                     n_hat[i, a] = -gsum_wg[i, a] / gnorm
#                 end
#             end
#         end
#     end

#     # ------------------------------------------------------------
#     # Curvature κ ≈ -∇·n_hat, fluid-only gather (kept, but softer limiter later)
#     # ------------------------------------------------------------
#     kappa = zeros(T, np)
#     foreach_point_neighbor(system, system, coordsA, coordsA, semi;
#                            points=each_integrated_particle(system)) do a,b,pos_diff,distance
#         if FS[a] || FS[b]
#             gW = smoothing_kernel_grad(system, pos_diff, distance, a)
#             acc = zero(T)
#             @inbounds for i in 1:dim
#                 acc += (n_hat[i, b] - n_hat[i, a]) * gW[i]
#             end
#             @inbounds kappa[a] -= acc
#         end
#     end
#     κ0 = max(T(1.2)/h, T(1e-6))  # relaxed (was 0.8/h)
#     p  = T(1.5)                  # softer slope (was 2)

#     # ------------------------------------------------------------
#     # Tangential XSPH smoothing on δv (two-buffer, normalized): gentler
#     # ------------------------------------------------------------
#     alpha_t = T(0.04)  # gentler (was 0.08)
#     dvt_src = zeros(T, dim, np) # tangential component of δv (source)
#     dvt_acc = zeros(T, dim, np) # gather increments
#     wsum    = zeros(T, np)

#     if alpha_t > 0
#         # build tangential source
#         @inbounds for a in 1:np
#             if FS[a]
#                 dotnv = zero(T)
#                 for i in 1:dim
#                     dotnv += n_hat[i, a] * system.cache.delta_v[i, a]
#                 end
#                 for i in 1:dim
#                     dvt_src[i, a] = system.cache.delta_v[i, a] - dotnv * n_hat[i, a]
#                 end
#             end
#         end

#         # gather smoothing (fluid-only, FS–FS)
#         foreach_point_neighbor(system, system, coordsA, coordsA, semi;
#                                points=each_integrated_particle(system)) do a,b,pos_diff,distance
#             if FS[a] && FS[b]
#                 Wab = smoothing_kernel(system, distance, a)
#                 @inbounds begin
#                     for i in 1:dim
#                         dvt_acc[i, a] += (dvt_src[i, b] - dvt_src[i, a]) * Wab
#                     end
#                     wsum[a] += Wab
#                 end
#             end
#         end

#         # finalize tangential smoothing back into dvt_src
#         @inbounds for a in 1:np
#             if FS[a] && isfinite(wsum[a]) && wsum[a] > epsT
#                 scale = alpha_t / wsum[a]
#                 for i in 1:dim
#                     dvt_src[i, a] += scale * dvt_acc[i, a]
#                 end
#             end
#         end
#     end

#     # ------------------------------------------------------------
#     # PASS 2: strongest two wall normals per particle (for corners)
#     # ------------------------------------------------------------
#     wall_g1 = zeros(T, dim, np)   # strongest wall normal proxy
#     wall_g2 = zeros(T, dim, np)   # second strongest
#     nrm1    = zeros(T, np)        # |g1|^2
#     nrm2    = zeros(T, np)        # |g2|^2

#     foreach_system(semi) do sysB
#         if !(sysB isa AbstractBoundarySystem)
#             return
#         end
#         if (u_ode === nothing) && (sysB !== system)
#             return
#         end
#         uB      = (u_ode === nothing) ? u : wrap_u(u_ode, sysB, semi)
#         coordsB = current_coordinates(uB, sysB)

#         # accumulate gradient sum for THIS wall system
#         gtmp = zeros(T, dim, np)
#         foreach_point_neighbor(system, sysB, coordsA, coordsB, semi;
#                                points=each_integrated_particle(system)) do a,b,pos_diff,distance
#             gW = smoothing_kernel_grad(system, pos_diff, distance, a)
#             @inbounds for i in 1:dim
#                 gtmp[i, a] += gW[i]
#             end
#         end

#         # update top-2 per particle
#         @inbounds for a in 1:np
#             wx = gtmp[1,a]; wy = dim>=2 ? gtmp[2,a] : zero(T); wz = dim==3 ? gtmp[3,a] : zero(T)
#             val = wx*wx + wy*wy + wz*wz
#             if val > nrm1[a] + epsT
#                 for i in 1:dim
#                     wall_g2[i,a] = wall_g1[i,a]
#                     wall_g1[i,a] = gtmp[i,a]
#                 end
#                 nrm2[a] = nrm1[a]
#                 nrm1[a] = val
#             elseif val > nrm2[a] + epsT
#                 for i in 1:dim
#                     wall_g2[i,a] = gtmp[i,a]
#                 end
#                 nrm2[a] = val
#             end
#         end
#     end

#     # ------------------------------------------------------------
#     # PASS 3: modify δv in place (kernel body)
#     # ------------------------------------------------------------
#     β_wall   = T(2)         # wall-proximity damping exponent
#     κ_corner = T(0.30)      # corner eigen-ratio threshold (2D)

#     @threaded semi for a in each_integrated_particle(system)
#         # --- 3.0: corner kill-switch (2D only)
#         if dim == 2
#             tr  = @inbounds S11[a] + S22[a]
#             det = @inbounds S11[a]*S22[a] - S12[a]*S12[a]
#             disc = max(tr*tr - 4*det, zero(T))
#             lam1 = T(0.5)*(tr + sqrt(disc))
#             lam2 = T(0.5)*(tr - sqrt(disc))
#             if lam1 > epsT && lam2/lam1 > κ_corner
#                 @inbounds for i in 1:dim
#                     system.cache.delta_v[i, a] = zero(T)
#                 end
#                 return
#             end
#         end

#         # --- 3.1: smooth wall-proximity damping
#         rw = @inbounds psi_wall[a] / (psi_wall[a] + psi_nog[a] + epsT)
#         scale_wall = (one(T) - clamp(rw, zero(T), one(T)))^β_wall
#         @inbounds for i in 1:dim
#             system.cache.delta_v[i, a] *= scale_wall
#         end

#         # --- 3.2: λ-cutoff (Sun'19): disable PST if λ is too small
#         λ = @inbounds psi_nog[a] / (smax_nog + epsT)
#         if λ < λ_min
#             @inbounds for i in 1:dim
#                 system.cache.delta_v[i, a] = zero(T)
#             end
#             return
#         end

#         # --- 3.3: enforce wall no-penetration (project out up to 2 wall normals)
#         n1 = nrm1[a]; n2 = nrm2[a]
#         if n1 > epsT
#             dot1 = zero(T)
#             @inbounds for i in 1:dim
#                 dot1 += system.cache.delta_v[i,a] * wall_g1[i,a]
#             end
#             scal1 = dot1 / (n1 + epsT)
#             @inbounds for i in 1:dim
#                 system.cache.delta_v[i,a] -= scal1 * wall_g1[i,a]
#             end
#         end
#         if n2 > epsT
#             # avoid re-projecting along near-colinear normals
#             dot2 = zero(T); g1g2 = zero(T)
#             @inbounds for i in 1:dim
#                 dot2 += system.cache.delta_v[i,a] * wall_g2[i,a]
#                 g1g2 += wall_g1[i,a]*wall_g2[i,a]
#             end
#             cos2 = (n1 > epsT) ? (g1g2*g1g2)/(n1*n2 + epsT) : zero(T)
#             if cos2 < T(0.99)
#                 scal2 = dot2 / (n2 + epsT)
#                 @inbounds for i in 1:dim
#                     system.cache.delta_v[i,a] -= scal2 * wall_g2[i,a]
#                 end
#             end
#         end

#         # --- 3.4: free-surface tangential-only (with sign test)
#         g2 = zero(T)
#         @inbounds for i in 1:dim
#             g2 += gsum_wg[i,a] * gsum_wg[i,a]
#         end
#         gnorm = sqrt(g2)
#         if FS[a] && gnorm > epsT
#             # apply only if (n · δu*) >= 0 (i.e., outward)
#             ndotu = zero(T)
#             @inbounds for i in 1:dim
#                 ndotu += (-gsum_wg[i,a] / gnorm) * system.cache.delta_v[i,a]
#             end
#             if ndotu >= 0
#                 @inbounds for i in 1:dim
#                     system.cache.delta_v[i,a] -= ndotu * (-gsum_wg[i,a] / (gnorm + epsT))
#                 end
#             end
#             # soft FS limiter based on support without ghosts (LESS damping)
#             s = @inbounds psi_nog[a] / (α_limit*smax_nog + epsT)
#             s = clamp(s, s_min, one(T))   # NEW: floor keeps some δu
#             @inbounds for i in 1:dim
#                 system.cache.delta_v[i,a] *= s
#             end
#         end

#         # --- 3.5: curvature-aware limiter (softer)
#         kap    = @inbounds abs(kappa[a])
#         f_curv = one(T) / (one(T) + (kap / (κ0 + epsT))^p)
#         @inbounds for i in 1:dim
#             system.cache.delta_v[i, a] *= f_curv
#         end

#         # --- 3.6: gently blend smoothed tangential δv (additive, not replacement)
#         if alpha_t > 0 && FS[a]
#             # add a small smoothed tangential increment, keep current normal part
#             dotnv = zero(T)
#             @inbounds for i in 1:dim
#                 dotnv += n_hat[i, a] * system.cache.delta_v[i, a]
#             end
#             @inbounds for i in 1:dim
#                 # δv ← current + (dvt_src - current_tangential) * η
#                 # but dvt_src already includes (I - nnᵀ)δv + smoothing; add small increment:
#                 inc = dvt_src[i, a] - (system.cache.delta_v[i, a] - dotnv * n_hat[i, a])
#                 system.cache.delta_v[i, a] += inc   # η=1 here since alpha_t is already small
#             end
#         end
#     end

#     # ------------------------------------------------------------
#     # PASS 4: sanitize NaNs/Infs (very cheap)
#     # ------------------------------------------------------------
#     @inbounds for a in 1:np
#         for i in 1:dim
#             v = system.cache.delta_v[i, a]
#             if !isfinite(v)
#                 system.cache.delta_v[i, a] = zero(T)
#             end
#         end
#     end

#     return system
# end

"""
    modify_shifting_at_free_surfaces!(system, u, semi, u_ode;
                                      λ_off::Real = 0.78,
                                      λ_on::Real  = 0.97,
                                      s_min::Real = 0.03,
                                      γ::Real     = 2.5)

Scale the **particle shifting velocity** `δv` near the free surface using a simple,
robust **neighbor–concentration ramp**. This attenuates shifting where particle
support is truncated (tips/crests/intersections) and leaves it unchanged in the bulk.

# How it works
For each particle `a`, we build a **support fullness**
`λ = N_a / N_ref ∈ [0,1]`, where `N_a` is the (fluid-phase) neighbor count found by
the neighbor loop and `N_ref = max_a N_a` (an interior reference).
We then compute a smooth ramp `s(λ)` and scale the current shifting velocity:

s_lin(λ) = clamp( (λ - λ_off) / (λ_on - λ_off), 0, 1 )
s(λ) = max(s_min, s_lin(λ))^γ
δv_a ← s(λ_a) * δv_a

markdown
Code kopieren

Thus:
- Very sparse support (`λ ≤ λ_off`) → `s ≈ s_min` (almost off).
- Full support (`λ ≥ λ_on`)       → `s ≈ 1`    (no attenuation).
- In-between (`λ_off < λ < λ_on`) → linear ramp (optionally made steeper by `γ`).

This strategy is commonly used in improved/enhanced PST variants to moderate
shifting at free surfaces without explicit normal/curvature estimation.

# Keyword parameters
- `λ_off` (default `0.78`): **Ramp start**. Support below this is considered
  “too sparse” for reliable shifting. Increasing `λ_off` **reduces** surface shifting
  (safer at thin tongues/crests), decreasing it **increases** surface shifting.
- `λ_on`  (default `0.97`): **Ramp end**. Support above this receives **full**
  shifting. Increasing `λ_on` makes full shifting harder to reach; decreasing it
  expands the region with full shifting.
- `s_min` (default `0.03`): **Floor** of the scaling factor to avoid freezing
  tangential reordering completely at the interface. Set to `0.0` to **disable
  shifting entirely** when `λ` is below `λ_off`. Larger `s_min` keeps a bit more
  motion but can promote peel-off/clustering if too high.
- `γ`     (default `2.5`): **Nonlinearity** of the ramp. `γ > 1` concentrates
  shifting in the bulk and damps it near the free surface; `γ < 1` does the
  opposite (more aggressive at the surface). Typical safe range is `1.5–3.0`.

# Practical tuning
- **Leading edge (dam-break tongue too mobile):** raise `λ_off` a little
  (e.g. `0.80`) or increase `γ` (e.g. `3.0`).
- **Surface clustering/banding:** raise `λ_off` or lower `s_min` (down to `0.0`
  if you prefer to fully switch off in very sparse regions). If clustering persists,
  consider adding a tiny outward **sign-clamp** (remove outward component of `δv`
  only when very close to the free surface) or a light δ-SPH density diffusion
  in your solver (outside this function).
- **More surface reordering:** lower `λ_off` slightly (e.g. `0.75`) or reduce `γ`.

# Notes & assumptions
- `N_a` is the neighbor count gathered by the provided neighbor traversal
  (as written, it counts neighbors from whatever systems you iterate in
  `foreach_system(semi)`; if you only want same-phase neighbors, limit that loop
  to `system` only).
- This routine **does not modify wall particles** and does not explicitly
  project along wall normals; it simply scales `δv` by local support fullness.
- The scaling is GPU- and threading-friendly: no global reductions beyond
  computing `N_ref = maximum(ncnt)` and only per-particle operations thereafter.

# Example
```julia
# safer at violent free surfaces (less shift at tips)
modify_shifting_at_free_surfaces!(sys, u, semi, u_ode; λ_off=0.80, λ_on=0.97, s_min=0.02, γ=3.0)

# slightly more surface ordering
modify_shifting_at_free_surfaces!(sys, u, semi, u_ode; λ_off=0.75, λ_on=0.95, s_min=0.05, γ=1.8)
"""
# function modify_shifting_at_free_surfaces!(system, u, semi, u_ode;
#                                            λ_off::Real = 0.78,
#                                            λ_on::Real  = 0.97,
#                                            s_min::Real = 0.03,
#                                            γ::Real     = 2.5)

function modify_shifting_at_free_surfaces!(system, u, semi, u_ode;
                                           λ_off::Real = 0.70, #2 78 #3 70 > 65 already rounds out too much # 70
                                           λ_on::Real  = 0.90, #2 95 #3 90 still fine #2 90
                                           s_min::Real = 0.0, #3 03 0.1 <- detaches 0.05 detaches from wall #2 0
                                           γ::Real     = 2.5) #2 2.5 #2 2.0 worse #2 3.0
    dim = ndims(system)
    epsT = eps(eltype(system.cache.delta_v))

    coordsA = current_coordinates(u, system)
    set_zero!(system.cache.nct)

    foreach_system(semi) do sysB
        uB      = wrap_u(u_ode, sysB, semi)
        coordsB = current_coordinates(uB, sysB)

        foreach_point_neighbor(system, sysB, coordsA, coordsB, semi;
                                points=each_integrated_particle(system)) do a,b,pos_diff,distance
                @inbounds system.cache.nct[a] += 1
        end
    end

    Nref = max(maximum(system.cache.nct), 1)

    inv_range = 1 / (λ_on - λ_off + epsT)

    @threaded semi for a in each_integrated_particle(system)
        λ = @inbounds system.cache.nct[a] / Nref     # 0..1 proxy for support fullness
        s = (λ - λ_off) * inv_range         # linear ramp [α0..α1] → [0..1]
        s = s < epsT ? 0 : (s > 1 ? 1 : s)
        s = max(s, s_min)                  # keep some ordering if desired
        sγ = s^γ
        for i in 1:dim
            @inbounds system.cache.delta_v[i, a] *= sγ
        end
    end

    return system
end


"""
Two-metric free-surface attenuation for PST (same-phase neighbors; walls only for FS mask).
Adds:
  • Tip clamp: extra ramp for very low same-phase support (leading edge).
  • Anti-clustering equalizer: downscale where N_a > local weighted average.

Tunables (safe defaults):
  λ_off=0.76, λ_on=0.965, γ=2.0, s_min=0.05,
  α_mask=0.92, clampλ=0.99,
  λ_tip0=0.52, λ_tip1=0.65, γ_tip=2.0, q_eq=2.0
"""
# function modify_shifting_at_free_surfaces!(system, u, semi, u_ode;
#     λ_off::Real=0.76, λ_on::Real=0.965, γ::Real=2.0, s_min::Real=0.05,
#     α_mask::Real=0.92, clampλ::Real=0.99,
#     λ_tip0::Real=0.52, λ_tip1::Real=0.65, γ_tip::Real=2.0, q_eq::Real=2.0)

#     T   = eltype(system.cache.delta_v)
#     dim = ndims(system)
#     np  = size(system.cache.delta_v, 2)
#     epsT = eps(T)

#     coordsA = current_coordinates(u, system)

#     # --- Pass 1: same-phase neighbor data -----------------------------------
#     N_fluid     = zeros(T, np)          # same-phase neighbor count
#     gsum_fluid  = zeros(T, dim, np)     # same-phase Σ∇W (for sign clamp)

#     foreach_point_neighbor(system, system, coordsA, coordsA, semi;
#                            points=each_integrated_particle(system)) do a,b,pos_diff,distance
#         @inbounds N_fluid[a] += one(T)
#         gWab = smoothing_kernel_grad(system, pos_diff, distance, a)
#         @inbounds for i in 1:dim
#             gsum_fluid[i, a] += gWab[i]
#         end
#     end

#     # --- Pass 2: N_all = same-phase + wall neighbors (FS mask only) ----------
#     N_all = copy(N_fluid)
#     foreach_system(semi) do sysB
#         (sysB isa AbstractBoundarySystem) || return
#         uB      = (u_ode === nothing) ? u : wrap_u(u_ode, sysB, semi)
#         coordsB = current_coordinates(uB, sysB)
#         foreach_point_neighbor(system, sysB, coordsA, coordsB, semi;
#                                points=each_integrated_particle(system)) do a,b,pos_diff,distance
#             @inbounds N_all[a] += one(T)
#         end
#     end

#     # --- Pass 3: local weighted average of same-phase neighbor count ----------
#     nbar_num = zeros(T, np)   # Σ W_ab * N_fluid[b]
#     nbar_den = zeros(T, np)   # Σ W_ab
#     foreach_point_neighbor(system, system, coordsA, coordsA, semi;
#                            points=each_integrated_particle(system)) do a,b,pos_diff,distance
#         Wab = smoothing_kernel(system, distance, a)
#         @inbounds begin
#             nbar_num[a] += Wab * N_fluid[b]
#             nbar_den[a] += Wab
#         end
#     end

#     # References from maxima (interior-like)
#     Nref_fluid = max(maximum(N_fluid), one(T))
#     Nref_all   = max(maximum(N_all),   one(T))

#     # Ramps/params
#     α0 = T(λ_off); α1 = T(λ_on); invr = one(T)/(α1-α0 + epsT)
#     αt0 = T(λ_tip0); αt1 = T(λ_tip1); invrt = one(T)/(αt1-αt0 + epsT)
#     gexp = T(γ); gtip = T(γ_tip); smin = T(s_min)
#     αmask = T(α_mask); clampλT = T(clampλ)
#     qeq = T(q_eq)

#     @threaded semi for a in each_integrated_particle(system)
#         # Free-surface mask uses all neighbors (fluid + walls)
#         λ_all = @inbounds N_all[a] / Nref_all
#         is_FS = λ_all < αmask
#         if !is_FS
#             return
#         end

#         # Same-phase fullness
#         λf = @inbounds N_fluid[a] / Nref_fluid

#         # Main FS ramp
#         s = (λf - α0) * invr
#         s = s < zero(T) ? zero(T) : (s > one(T) ? one(T) : s)

#         # Tip clamp (very low same-phase support → heavily attenuate)
#         s_tip = (λf - αt0) * invrt
#         s_tip = s_tip < zero(T) ? zero(T) : (s_tip > one(T) ? one(T) : s_tip)

#         s_total = max(smin, (s^gexp) * (s_tip^gtip))

#         @inbounds for i in 1:dim
#             system.cache.delta_v[i, a] *= s_total
#         end

#         # Anti-clustering equalizer vs local weighted average (same-phase only)
#         nbar = @inbounds nbar_num[a] / (nbar_den[a] + epsT)
#         if @inbounds N_fluid[a] > nbar
#             feq = (nbar / (@inbounds N_fluid[a] + epsT))^qeq
#             @inbounds for i in 1:dim
#                 system.cache.delta_v[i, a] *= feq
#             end
#         end

#         # Tiny outward sign clamp close to FS (prevents peel-off)
#         if λ_all < clampλT
#             g2 = zero(T)
#             @inbounds for i in 1:dim
#                 g2 += gsum_fluid[i, a]^2
#             end
#             if g2 > T(1e2)*epsT
#                 invg = inv(sqrt(g2))
#                 ndot = zero(T)
#                 @inbounds for i in 1:dim
#                     ndot += system.cache.delta_v[i, a] * (-gsum_fluid[i, a] * invg)
#                 end
#                 if ndot > 0
#                     @inbounds for i in 1:dim
#                         system.cache.delta_v[i, a] -= ndot * (-gsum_fluid[i, a] * invg)
#                     end
#                 end
#             end
#         end
#     end

#     return system
# end

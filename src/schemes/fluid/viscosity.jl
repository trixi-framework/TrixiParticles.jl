# Unpack the neighboring systems viscosity to dispatch on the viscosity type.
# This function is only necessary to allow `nothing` as viscosity.
# Otherwise, we could just apply the viscosity as a function directly.
@propagate_inbounds function dv_viscosity(particle_system, neighbor_system,
                                          v_particle_system, v_neighbor_system,
                                          particle, neighbor, pos_diff, distance,
                                          sound_speed, m_a, m_b, rho_a, rho_b, grad_kernel)
    viscosity = viscosity_model(particle_system, neighbor_system)

    return dv_viscosity(viscosity, particle_system, neighbor_system,
                        v_particle_system, v_neighbor_system,
                        particle, neighbor, pos_diff, distance,
                        sound_speed, m_a, m_b, rho_a, rho_b, grad_kernel)
end

@propagate_inbounds function dv_viscosity(viscosity, particle_system, neighbor_system,
                                          v_particle_system, v_neighbor_system,
                                          particle, neighbor, pos_diff, distance,
                                          sound_speed, m_a, m_b, rho_a, rho_b, grad_kernel)
    return viscosity(particle_system, neighbor_system,
                     v_particle_system, v_neighbor_system,
                     particle, neighbor, pos_diff, distance,
                     sound_speed, m_a, m_b, rho_a, rho_b, grad_kernel)
end

@inline function dv_viscosity(viscosity::Nothing, particle_system, neighbor_system,
                              v_particle_system, v_neighbor_system,
                              particle, neighbor, pos_diff, distance,
                              sound_speed, m_a, m_b, rho_a, rho_b, grad_kernel)
    return zero(pos_diff)
end

@doc raw"""
    ArtificialViscosityMonaghan(; alpha, beta=0.0, epsilon=0.01)

Artificial viscosity by Monaghan ([Monaghan1992](@cite), [Monaghan1989](@cite)).

See [`Viscosity`](@ref viscosity_sph) for an overview and comparison of implemented viscosity models.

# Keywords
- `alpha`: A value of `0.02` is usually used for most simulations. For a relation with the
           kinematic viscosity, see description above.
- `beta=0.0`: A value of `0.0` works well for most fluid simulations and simulations
              with shocks of moderate strength. In simulations where the Mach number can be
              very high, eg. astrophysical calculation, good results can be obtained by
              choosing a value of `beta=2.0` and `alpha=1.0`.
- `epsilon=0.01`: Parameter to prevent singularities.
"""
struct ArtificialViscosityMonaghan{ELTYPE}
    alpha   :: ELTYPE
    beta    :: ELTYPE
    epsilon :: ELTYPE

    function ArtificialViscosityMonaghan(; alpha, beta=0.0, epsilon=0.01)
        new{typeof(alpha)}(alpha, beta, epsilon)
    end
end

@doc raw"""
    ViscosityMorris(; nu, epsilon=0.01)

Viscosity by [Morris (1997)](@cite Morris1997) also used by [Fourtakas (2019)](@cite Fourtakas2019).

See [`Viscosity`](@ref viscosity_sph) for an overview and comparison of implemented viscosity models.

# Keywords
- `nu`: Kinematic viscosity
- `epsilon=0.01`: Parameter to prevent singularities
"""
struct ViscosityMorris{ELTYPE}
    nu::ELTYPE
    epsilon::ELTYPE

    function ViscosityMorris(; nu, epsilon=0.01)
        new{typeof(nu)}(nu, epsilon)
    end
end

function kinematic_viscosity(system, viscosity::ViscosityMorris, smoothing_length,
                             sound_speed)
    return viscosity.nu
end

@propagate_inbounds function (viscosity::Union{ArtificialViscosityMonaghan,
                                               ViscosityMorris})(particle_system,
                                                                 neighbor_system,
                                                                 v_particle_system,
                                                                 v_neighbor_system,
                                                                 particle, neighbor,
                                                                 pos_diff, distance,
                                                                 sound_speed,
                                                                 m_a, m_b, rho_a, rho_b,
                                                                 grad_kernel)
    rho_mean = (rho_a + rho_b) / 2

    v_a = viscous_velocity(v_particle_system, particle_system, particle)
    v_b = viscous_velocity(v_neighbor_system, neighbor_system, neighbor)
    v_diff = v_a - v_b

    smoothing_length_particle = smoothing_length(particle_system, particle)
    smoothing_length_neighbor = smoothing_length(particle_system, neighbor)
    smoothing_length_average = (smoothing_length_particle + smoothing_length_neighbor) / 2

    nu_a = kinematic_viscosity(particle_system,
                               viscosity_model(neighbor_system, particle_system),
                               smoothing_length_particle, sound_speed)
    nu_b = kinematic_viscosity(neighbor_system,
                               viscosity_model(particle_system, neighbor_system),
                               smoothing_length_neighbor, sound_speed)

    pi_ab = viscosity(sound_speed, v_diff, pos_diff, distance, rho_mean, rho_a, rho_b,
                      smoothing_length_average, grad_kernel, nu_a, nu_b)

    return m_b * pi_ab
end

@inline function (viscosity::ArtificialViscosityMonaghan)(c, v_diff, pos_diff, distance,
                                                          rho_mean, rho_a, rho_b, h,
                                                          grad_kernel, nu_a, nu_b)
    (; alpha, beta, epsilon) = viscosity

    # v_ab ⋅ r_ab
    vr = dot(v_diff, pos_diff)

    # Monaghan 2005 p. 1741 (doi: 10.1088/0034-4885/68/8/r01):
    # "In the case of shock tube problems, it is usual to turn the viscosity on for
    # approaching particles and turn it off for receding particles. In this way, the
    # viscosity is used for shocks and not rarefactions."
    if vr < 0
        mu = h * vr / (distance^2 + epsilon * h^2)
        return (alpha * c * mu + beta * mu^2) / rho_mean * grad_kernel
    end

    return zero(v_diff)
end

@inline function (viscosity::ViscosityMorris)(c, v_diff, pos_diff, distance, rho_mean,
                                              rho_a, rho_b, h, grad_kernel, nu_a,
                                              nu_b)
    epsilon = viscosity.epsilon

    mu_a = nu_a * rho_a
    mu_b = nu_b * rho_b

    return (mu_a + mu_b) / (rho_a * rho_b) * dot(pos_diff, grad_kernel) /
           (distance^2 + epsilon * h^2) * v_diff
end

# See, e.g.,
# Joseph J. Monaghan. "Smoothed Particle Hydrodynamics".
# In: Reports on Progress in Physics (2005), pages 1703-1759.
# [doi: 10.1088/0034-4885/68/8/r01](http://dx.doi.org/10.1088/0034-4885/68/8/R01)
function kinematic_viscosity(system, viscosity::ArtificialViscosityMonaghan,
                             smoothing_length, sound_speed)
    (; alpha) = viscosity

    return alpha * smoothing_length * sound_speed / (2 * ndims(system) + 4)
end

@doc raw"""
    ViscosityAdami(; nu, epsilon=0.01)

Viscosity by [Adami (2012)](@cite Adami2012).

See [`Viscosity`](@ref viscosity_sph) for an overview and comparison of implemented viscosity models.

# Keywords
- `nu`: Kinematic viscosity
- `epsilon=0.01`: Parameter to prevent singularities
"""
struct ViscosityAdami{ELTYPE}
    nu::ELTYPE
    epsilon::ELTYPE

    function ViscosityAdami(; nu, epsilon=0.01)
        new{typeof(nu)}(nu, epsilon)
    end
end

function adami_viscosity_force(smoothing_length_average, pos_diff, distance, grad_kernel,
                               m_a, m_b, rho_a, rho_b, v_diff, nu_a, nu_b, epsilon)
    eta_a = nu_a * rho_a
    eta_b = nu_b * rho_b

    eta_tilde = 2 * (eta_a * eta_b) / (eta_a + eta_b)

    tmp = eta_tilde / (distance^2 + epsilon * smoothing_length_average^2)

    volume_a = m_a / rho_a
    volume_b = m_b / rho_b

    # This formulation was introduced by Hu and Adams (2006). https://doi.org/10.1016/j.jcp.2005.09.001
    # They argued that the formulation is more flexible because of the possibility to formulate
    # different inter-particle averages or to assume different inter-particle distributions.
    # Ramachandran (2019) and Adami (2012) use this formulation also for the pressure acceleration.
    #
    # TODO: Is there a better formulation to discretize the Laplace operator?
    # Because when using this formulation for the pressure acceleration, it is not
    # energy conserving.
    # See issue: https://github.com/trixi-framework/TrixiParticles.jl/issues/394
    visc = (volume_a^2 + volume_b^2) * dot(grad_kernel, pos_diff) * tmp / m_a

    return visc .* v_diff
end

@inline function (viscosity::ViscosityAdami)(particle_system, neighbor_system,
                                             v_particle_system, v_neighbor_system,
                                             particle, neighbor, pos_diff,
                                             distance, sound_speed, m_a, m_b,
                                             rho_a, rho_b, grad_kernel)
    epsilon = viscosity.epsilon

    smoothing_length_particle = smoothing_length(particle_system, particle)
    smoothing_length_neighbor = smoothing_length(particle_system, neighbor)
    smoothing_length_average = (smoothing_length_particle + smoothing_length_neighbor) / 2

    nu_a = kinematic_viscosity(particle_system,
                               viscosity_model(neighbor_system, particle_system),
                               smoothing_length_particle, sound_speed)
    nu_b = kinematic_viscosity(neighbor_system,
                               viscosity_model(particle_system, neighbor_system),
                               smoothing_length_neighbor, sound_speed)

    v_a = viscous_velocity(v_particle_system, particle_system, particle)
    v_b = viscous_velocity(v_neighbor_system, neighbor_system, neighbor)
    v_diff = v_a - v_b

    return adami_viscosity_force(smoothing_length_average, pos_diff, distance, grad_kernel,
                                 m_a, m_b, rho_a, rho_b, v_diff, nu_a, nu_b, epsilon)
end

function kinematic_viscosity(system, viscosity::ViscosityAdami, smoothing_length,
                             sound_speed)
    return viscosity.nu
end

@propagate_inbounds function viscous_velocity(v, system, particle)
    return current_velocity(v, system, particle)
end

@doc raw"""
    ViscosityAdamiSGS(; nu, C_S=0.1, epsilon=0.01)

Viscosity model that extends the standard [Adami formulation](@ref ViscosityAdami)
by incorporating a subgrid-scale (SGS) eddy viscosity via a Smagorinsky-type [Smagorinsky (1963)](@cite Smagorinsky1963) closure.
The effective kinematic viscosity is defined as

```math
\nu_{\text{eff}} = \nu_{\text{std}} + \nu_{\text{SGS}},
```

with

```math
\nu_{\text{SGS}} = (C_S h)^2 |S|,
```

and an approximation for the strain rate magnitude given by

```math
|S| \approx \frac{\|v_{ab}\|}{\|r_{ab}\| + \epsilon},
```

where:
- ``C_S`` is the Smagorinsky constant (typically 0.1 to 0.2),
- ``h`` is the local smoothing length.

The effective dynamic viscosities are then computed as
```math
\eta_{a,\text{eff}} = \rho_a\, \nu_{\text{eff}},
```
and averaged as
```math
\bar{\eta}_{ab} = \frac{2 \eta_{a,\text{eff}} \eta_{b,\text{eff}}}{\eta_{a,\text{eff}}+\eta_{b,\text{eff}}}.
```

This model is appropriate for turbulent flows where unresolved scales contribute additional dissipation.

# Keywords
- `nu`:      Standard kinematic viscosity.
- `C_S`:     Smagorinsky constant.
- `epsilon=0.01`: Parameter to prevent singularities
"""
struct ViscosityAdamiSGS{ELTYPE}
    nu      :: ELTYPE      # kinematic viscosity [e.g., 1e-6 m²/s]
    C_S     :: ELTYPE     # Smagorinsky constant [e.g., 0.1-0.2]
    epsilon :: ELTYPE # Epsilon for singularity prevention [e.g., 0.001]
end

ViscosityAdamiSGS(; nu, C_S=0.1, epsilon=0.001) = ViscosityAdamiSGS(nu, C_S, epsilon)

@propagate_inbounds function (viscosity::ViscosityAdamiSGS)(particle_system,
                                                            neighbor_system,
                                                            v_particle_system,
                                                            v_neighbor_system,
                                                            particle, neighbor, pos_diff,
                                                            distance, sound_speed, m_a, m_b,
                                                            rho_a, rho_b, grad_kernel)
    epsilon = viscosity.epsilon

    smoothing_length_particle = smoothing_length(particle_system, particle)
    smoothing_length_neighbor = smoothing_length(particle_system, neighbor)
    smoothing_length_average = (smoothing_length_particle + smoothing_length_neighbor) / 2

    nu_a = kinematic_viscosity(particle_system,
                               viscosity_model(neighbor_system, particle_system),
                               smoothing_length_particle, sound_speed)
    nu_b = kinematic_viscosity(neighbor_system,
                               viscosity_model(particle_system, neighbor_system),
                               smoothing_length_neighbor, sound_speed)

    v_a = viscous_velocity(v_particle_system, particle_system, particle)
    v_b = viscous_velocity(v_neighbor_system, neighbor_system, neighbor)
    v_diff = v_a - v_b

    # ------------------------------------------------------------------------------
    # SGS part: Compute the subgrid-scale eddy viscosity.
    # ------------------------------------------------------------------------------
    # In classical LES [Lilly (1967)](@cite Lilly1967) the Smagorinsky model defines:
    #   ν_SGS = (C_S Δ)^2 |S|,
    # where |S| is the norm of the strain-rate tensor Sᵢⱼ = ½(∂ᵢvⱼ+∂ⱼvᵢ).
    #
    # In SPH, one could compute ∂ᵢvⱼ via kernel gradients, but this is costly.
    # A common low-order surrogate is to approximate the strain‐rate magnitude by a
    # finite difference along each particle pair:
    #
    #   |S| ≈ ‖v_ab‖ / (‖r_ab‖ + δ),
    #
    # where δ regularizes the denominator to avoid singularities when particles are very close.
    #
    # This yields:
    #   S_mag = norm(v_diff) / (distance + ε)
    #
    # and then the Smagorinsky eddy viscosity:
    #   ν_SGS = (C_S * h̄)^2 * S_mag.
    #
    S_mag = norm(v_diff) / (distance + epsilon)
    nu_SGS = (viscosity.C_S * smoothing_length_average)^2 * S_mag

    # Effective kinematic viscosity is the sum of the standard and SGS parts.
    nu_a = nu_a + nu_SGS
    nu_b = nu_b + nu_SGS

    return adami_viscosity_force(smoothing_length_average, pos_diff, distance, grad_kernel,
                                 m_a, m_b, rho_a, rho_b, v_diff, nu_a, nu_b, epsilon)
end

function kinematic_viscosity(system, viscosity::ViscosityAdamiSGS, smoothing_length,
                             sound_speed)
    return viscosity.nu
end

@doc raw"""
    ViscosityMorrisSGS(; nu, C_S=0.1, epsilon=0.001)

Subgrid-scale (SGS) viscosity model based on the formulation by [Morris (1997)](@cite Morris1997),
by incorporating a subgrid-scale (SGS) eddy viscosity via a Smagorinsky-type [Smagorinsky (1963)](@cite Smagorinsky1963) closure.
The effective kinematic viscosity is defined as

```math
\nu_{\text{eff}} = \nu_{\text{std}} + \nu_{\text{SGS}},
```

with

```math
\nu_{\text{SGS}} = (C_S h)^2 |S|,
```

and an approximation for the strain rate magnitude given by

```math
|S| \approx \frac{\|v_{ab}\|}{\|r_{ab}\| + \epsilon},
```

where:
- ``C_S`` is the Smagorinsky constant (typically 0.1 to 0.2),
- ``h`` is the local smoothing length.

The effective dynamic viscosities are then computed as
```math
\eta_{a,\text{eff}} = \rho_a\, \nu_{\text{eff}},
```
and averaged as
```math
\bar{\eta}_{ab} = \frac{2 \eta_{a,\text{eff}} \eta_{b,\text{eff}}}{\eta_{a,\text{eff}}+\eta_{b,\text{eff}}}.
```

This model is appropriate for turbulent flows where unresolved scales contribute additional dissipation.

# Keywords
- `nu`:      Standard kinematic viscosity.
- `C_S`:     Smagorinsky constant.
- `epsilon=0.01`: Parameter to prevent singularities
"""
struct ViscosityMorrisSGS{ELTYPE}
    nu::ELTYPE      # kinematic viscosity [e.g., 1e-6 m²/s]
    C_S::ELTYPE     # Smagorinsky constant [e.g., 0.1-0.2]
    epsilon::ELTYPE # Epsilon for singularity prevention [e.g., 0.001]
end

ViscosityMorrisSGS(; nu, C_S=0.1, epsilon=0.001) = ViscosityMorrisSGS(nu, C_S, epsilon)

@propagate_inbounds function (viscosity::ViscosityMorrisSGS)(particle_system,
                                                             neighbor_system,
                                                             v_particle_system,
                                                             v_neighbor_system,
                                                             particle, neighbor, pos_diff,
                                                             distance, sound_speed, m_a,
                                                             m_b, rho_a, rho_b, grad_kernel)
    epsilon = viscosity.epsilon

    smoothing_length_particle = smoothing_length(particle_system, particle)
    smoothing_length_neighbor = smoothing_length(particle_system, neighbor)
    smoothing_length_average = (smoothing_length_particle + smoothing_length_neighbor) / 2

    nu_a = kinematic_viscosity(particle_system,
                               viscosity_model(neighbor_system, particle_system),
                               smoothing_length_particle, sound_speed)
    nu_b = kinematic_viscosity(neighbor_system,
                               viscosity_model(particle_system, neighbor_system),
                               smoothing_length_neighbor, sound_speed)

    v_a = viscous_velocity(v_particle_system, particle_system, particle)
    v_b = viscous_velocity(v_neighbor_system, neighbor_system, neighbor)
    v_diff = v_a - v_b

    # SGS part: Compute the subgrid-scale eddy viscosity.
    # See comments above for `ViscosityAdamiSGS`.
    S_mag = norm(v_diff) / (distance + epsilon)
    nu_SGS = (viscosity.C_S * smoothing_length_average)^2 * S_mag

    # Effective viscosities include the SGS term.
    nu_a_eff = nu_a + nu_SGS
    nu_b_eff = nu_b + nu_SGS

    # For the Morris model, dynamic viscosities are:
    mu_a = nu_a_eff * rho_a
    mu_b = nu_b_eff * rho_b

    force_Morris = (mu_a + mu_b) / (rho_a * rho_b) * (dot(pos_diff, grad_kernel)) /
                   (distance^2 + epsilon * smoothing_length_average^2) * v_diff
    return m_b * force_Morris
end

function kinematic_viscosity(system, viscosity::ViscosityMorrisSGS, smoothing_length,
                             sound_speed)
    return viscosity.nu
end

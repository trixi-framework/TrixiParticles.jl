
# Unpack the neighboring systems viscosity to dispatch on the viscosity type
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

Artificial viscosity by Monaghan ([Monaghan1992](@cite), [Monaghan1989](@cite)), given by
```math
\Pi_{ab} =
\begin{cases}
    -(\alpha c \mu_{ab} + \beta \mu_{ab}^2) / \bar{\rho}_{ab} & \text{if } v_{ab} \cdot r_{ab} < 0, \\
    0 & \text{otherwise}
\end{cases}
```
with
```math
\mu_{ab} = \frac{h v_{ab} \cdot r_{ab}}{\Vert r_{ab} \Vert^2 + \epsilon h^2},
```
where ``\alpha, \beta, \epsilon`` are parameters, ``c`` is the speed of sound, ``h`` is the smoothing length,
``r_{ab} = r_a - r_b`` is the difference of the coordinates of particles ``a`` and ``b``,
``v_{ab} = v_a - v_b`` is the difference of their velocities,
and ``\bar{\rho}_{ab}`` is the arithmetic mean of their densities.

Note that ``\alpha`` needs to adjusted for different resolutions to maintain a specific Reynolds Number.
To do so, [Monaghan (2005)](@cite Monaghan2005) defined an equivalent effective physical kinematic viscosity ``\nu`` by
```math
    \nu = \frac{\alpha h c }{2d + 4},
```
where ``d`` is the dimension.

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

To the force ``f_{ab}`` between two particles ``a`` and ``b`` due to pressure gradients,
an additional force term ``\tilde{f}_{ab}`` is added with
```math
\tilde{f}_{ab} = m_a m_b \frac{(\mu_a + \mu_b) r_{ab} \cdot \nabla W_{ab}}{\rho_a \rho_b (\Vert r_{ab} \Vert^2 + \epsilon h^2)} v_{ab},
```
where ``\mu_a = \rho_a \nu`` and ``\mu_b = \rho_b \nu`` denote the dynamic viscosity
of particle ``a`` and ``b`` respectively, and ``\nu`` is the kinematic viscosity.

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

    nu_a = kinematic_viscosity(particle_system,
                               viscosity_model(neighbor_system, particle_system),
                               smoothing_length_particle, sound_speed)
    nu_b = kinematic_viscosity(neighbor_system,
                               viscosity_model(particle_system, neighbor_system),
                               smoothing_length_neighbor, sound_speed)

    smoothing_length_average = (smoothing_length_particle + smoothing_length_neighbor) / 2
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
The viscous interaction is calculated with the shear force for incompressible flows given by
```math
f_{ab} = \sum_w \bar{\eta}_{ab} \left( V_a^2 + V_b^2 \right) \frac{v_{ab}}{||r_{ab}||^2+\epsilon h_{ab}^2}  \nabla W_{ab} \cdot r_{ab},
```
where ``r_{ab} = r_a - r_b`` is the difference of the coordinates of particles ``a`` and ``b``,
``v_{ab} = v_a - v_b`` is the difference of their velocities, ``h`` is the smoothing length and ``V`` is the particle volume.
The parameter ``\epsilon`` prevents singularities (see [Ramachandran (2019)](@cite Ramachandran2019)).
The inter-particle-averaged shear stress  is
```math
    \bar{\eta}_{ab} =\frac{2 \eta_a \eta_b}{\eta_a + \eta_b},
```
where ``\eta_a = \rho_a \nu_a`` with ``\nu`` as the kinematic viscosity.

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
                               m_a,
                               m_b, rho_a, rho_b, v_diff, nu_a, nu_b, epsilon)
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

Viscosity model that extends the standard Adami formulation by incorporating a subgrid-scale (SGS)
eddy viscosity via a Smagorinsky-type closure. The effective kinematic viscosity is defined as

```
\nu_{\mathrm{eff}} = \nu_{\\mathrm{std}} + \nu_{\\mathrm{SGS}},
```

with

```
\nu_{\mathrm{SGS}} = (C_S * h)^2 * |S|,
```

and an approximation for the strain rate magnitude given by

```
|S| \approx \frac{\|v_a - v_b\|}{\|r_a - r_b\| + \epsilon},
```

where:
- $C_S$ is the Smagorinsky constant (typically 0.1 to 0.2),
- $h$ is the local smoothing length, and

The effective dynamic viscosities are then computed as
```
\eta_{a,\mathrm{eff}} = \rho_a\, \nu_{\mathrm{eff}},
```
and averaged as
```
\bar{\eta}_{ab} = \frac{2 \eta_{a,\mathrm{eff}} \eta_{b,\mathrm{eff}}}{\eta_{a,\mathrm{eff}}+\eta_{b,\mathrm{eff}}}.
```

This model is appropriate for turbulent flows where unresolved scales contribute additional dissipation.

# Keywords
- `nu`:      Standard kinematic viscosity.
- `C_S`:     Smagorinsky constant.
- `epsilon`: Epsilon for singularity prevention [e.g., 0.001]
"""
struct ViscosityAdamiSGS{ELTYPE}
    nu::ELTYPE      # Standard (molecular) kinematic viscosity [e.g., 1e-6 m²/s]
    C_S::ELTYPE     # Smagorinsky constant [e.g., 0.1-0.2]
    epsilon::ELTYPE # Epsilon for singularity prevention [e.g., 0.001]
end

# Convenient constructor with default values for C_S and epsilon.
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
    # Estimate the strain rate magnitude |S| (rough approximation)
    S_mag = norm(v_diff) / (distance + epsilon)
    nu_SGS = (C_S * smoothing_length_average)^2 * S_mag

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
extended with a Smagorinsky-type eddy viscosity term for modeling turbulent flows.

The acceleration on particle `a` due to viscosity interaction with particle `b` is calculated as:
```math
\frac{d v_a}{dt} = \sum_b m_b \frac{\mu_{a,\mathrm{eff}} + \mu_{b,\mathrm{eff}}}{\rho_a \rho_b} \frac{r_{ab} \cdot \nabla_a W_{ab}}{r_{ab}^2 + \epsilon h_{ab}^2} v_{ab}
where $v_{ab} = v_a - v_b$, $r_{ab} = r_a - r_b$, $r_{ab} = \|r_{ab}\|$, $h_{ab} = (h_a + h_b)/2$, $W_{ab}$ is the smoothing kernel, $\rho$ is density, $m$ is mass, and $\epsilon$ is a regularization parameter.

The effective dynamic viscosity $\mu_{i,\mathrm{eff}}$ for each particle `i` (a or b) includes both the standard molecular viscosity and an SGS eddy viscosity contribution:
```math
\mu_{i,\mathrm{eff}} = \rho_i \nu_{i,\mathrm{eff}} = \rho_i (\nu_{\mathrm{std}} + \nu_{i,\mathrm{SGS}})
```
The standard kinematic viscosity $\nu_{\mathrm{std}}$ is provided by the `nu` parameter. The SGS kinematic viscosity is calculated using the Smagorinsky model:
```math
\nu_{i,\mathrm{SGS}} = (C_S h_{ab})^2 |\bar{S}_{ab}|
```
where $C_S$ is the Smagorinsky constant. This implementation uses a simplified pairwise approximation for the strain rate tensor magnitude $|\bar{S}_{ab}|$:
```math
|\bar{S}_{ab}| \approx \frac{\|v_a - v_b\|}{\|r_a - r_b\| + \epsilon}
```

This model is appropriate for turbulent flows where unresolved scales contribute additional dissipation.

# Keywords
- `nu`:      Standard kinematic viscosity.
- `C_S`:     Smagorinsky constant.
- `epsilon`: Epsilon for singularity prevention [e.g., 0.001]
"""
ViscosityMorrisSGS(; nu, C_S=0.1, epsilon=0.001) = ViscosityMorrisSGS(nu, C_S, epsilon)

@propagate_inbounds function (viscosity::ViscosityMorrisSGS)(particle_system,
                                                             neighbor_system,
                                                             v_particle_system,
                                                             v_neighbor_system,
                                                             particle, neighbor, pos_diff,
                                                             distance, sound_speed, m_a,
                                                             m_b,
                                                             rho_a, rho_b, grad_kernel)
    epsilon_val = viscosity.epsilon
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
    # Estimate the strain rate magnitude |S| (rough approximation)
    S_mag = norm(v_diff) / (distance + epsilon)
    nu_SGS = (C_S * smoothing_length_average)^2 * S_mag

    # Effective viscosities include the SGS term.
    nu_a_eff = nu_a + nu_SGS
    nu_b_eff = nu_b + nu_SGS

    # For the Morris model, dynamic viscosities are:
    mu_a = nu_a_eff * rho_a
    mu_b = nu_b_eff * rho_b

    force_Morris = (mu_a + mu_b) / (rho_a * rho_b) * (dot(pos_diff, grad_kernel)) /
                   (distance^2 + epsilon_val * smoothing_length_average^2) * v_diff
    return m_b * force_Morris
end

function kinematic_viscosity(system, viscosity::ViscosityMorrisSGS, smoothing_length,
                             sound_speed)
    return viscosity.nu
end

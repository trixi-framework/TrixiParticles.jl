
# Unpack the neighboring systems viscosity to dispatch on the viscosity type
function dv_viscosity(particle_system, neighbor_system,
                      v_particle_system, v_neighbor_system,
                      particle, neighbor, pos_diff, distance,
                      sound_speed, m_a, m_b, rho_a, rho_b, grad_kernel)
    viscosity = viscosity_model(neighbor_system)

    return dv_viscosity(viscosity, particle_system, neighbor_system,
                        v_particle_system, v_neighbor_system,
                        particle, neighbor, pos_diff, distance,
                        sound_speed, m_a, m_b, rho_a, rho_b, grad_kernel)
end

function dv_viscosity(particle_system, neighbor_system::OpenBoundarySPHSystem,
                      v_particle_system, v_neighbor_system,
                      particle, neighbor, pos_diff, distance,
                      sound_speed, m_a, m_b, rho_a, rho_b, grad_kernel)
    # No viscosity in the open boundary system. Use viscosity of the fluid system.
    viscosity = viscosity_model(particle_system)

    return dv_viscosity(viscosity, particle_system, neighbor_system,
                        v_particle_system, v_neighbor_system,
                        particle, neighbor, pos_diff, distance,
                        sound_speed, m_a, m_b, rho_a, rho_b, grad_kernel)
end

function dv_viscosity(viscosity, particle_system, neighbor_system,
                      v_particle_system, v_neighbor_system,
                      particle, neighbor, pos_diff, distance,
                      sound_speed, m_a, m_b, rho_a, rho_b, grad_kernel)
    return viscosity(particle_system, neighbor_system,
                     v_particle_system, v_neighbor_system,
                     particle, neighbor, pos_diff, distance,
                     sound_speed, m_a, m_b, rho_a, rho_b, grad_kernel)
end

function dv_viscosity(viscosity::Nothing, particle_system, neighbor_system,
                      v_particle_system, v_neighbor_system,
                      particle, neighbor, pos_diff, distance,
                      sound_speed, m_a, m_b, rho_a, rho_b, grad_kernel)
    return zero(pos_diff)
end

@doc raw"""
    ArtificialViscosityMonaghan(; alpha, beta=0.0, epsilon=0.01)

Artificial viscosity by Monaghan (Monaghan 1992, Monaghan 1989), given by
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
To do so, Monaghan (Monaghan 2005) defined an equivalent effective physical kinematic viscosity ``\nu`` by
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

## References
- Joseph J. Monaghan. "Smoothed Particle Hydrodynamics".
  In: Annual Review of Astronomy and Astrophysics 30.1 (1992), pages 543-574.
  [doi: 10.1146/ANNUREV.AA.30.090192.002551](https://doi.org/10.1146/ANNUREV.AA.30.090192.002551)
- Joseph J. Monaghan. "Smoothed Particle Hydrodynamics".
  In: Reports on Progress in Physics (2005), pages 1703-1759.
  [doi: 10.1088/0034-4885/68/8/r01](http://dx.doi.org/10.1088/0034-4885/68/8/R01)
- Joseph J. Monaghan. "On the Problem of Penetration in Particle Methods".
  In: Journal of Computational Physics 82.1, pages 1–15.
  [doi: 10.1016/0021-9991(89)90032-6](https://doi.org/10.1016/0021-9991(89)90032-6)
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

Viscosity by Morris et al. (1997).

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

## References
- Joseph P. Morris, Patrick J. Fox, Yi Zhu.
  "Modeling Low Reynolds Number Incompressible Flows Using SPH".
  In: Journal of Computational Physics, Volume 136, Issue 1 (1997), pages 214--226.
  [doi: doi.org/10.1006/jcph.1997.5776](https://doi.org/10.1006/jcph.1997.5776)
- Georgios Fourtakas, Jose M. Dominguez, Renato Vacondio, Benedict D. Rogers.
  "Local uniform stencil (LUST) boundary condition for arbitrary
  3-D boundaries in parallel smoothed particle hydrodynamics (SPH) models".
  In: Computers & Fluids, Volume 190 (2019), pages 346--361.
  [doi: 10.1016/j.compfluid.2019.06.009](https://doi.org/10.1016/j.compfluid.2019.06.009)
"""
struct ViscosityMorris{ELTYPE}
    nu::ELTYPE
    epsilon::ELTYPE

    function ViscosityMorris(; nu, epsilon=0.01)
        new{typeof(nu)}(nu, epsilon)
    end
end

function kinematic_viscosity(system, viscosity::ViscosityMorris)
    return viscosity.nu
end

@inline function (viscosity::Union{ArtificialViscosityMonaghan,
                                   ViscosityMorris})(particle_system, neighbor_system,
                                                     v_particle_system, v_neighbor_system,
                                                     particle, neighbor, pos_diff,
                                                     distance, sound_speed, m_a, m_b,
                                                     rho_a, rho_b, grad_kernel)
    (; smoothing_length) = particle_system

    rho_mean = 0.5 * (rho_a + rho_b)

    v_a = viscous_velocity(v_particle_system, particle_system, particle)
    v_b = viscous_velocity(v_neighbor_system, neighbor_system, neighbor)
    v_diff = v_a - v_b

    pi_ab = viscosity(sound_speed, v_diff, pos_diff, distance, rho_mean, rho_a, rho_b,
                      smoothing_length, grad_kernel)

    return m_b * pi_ab
end

@inline function (viscosity::ArtificialViscosityMonaghan)(c, v_diff, pos_diff, distance,
                                                          rho_mean, rho_a, rho_b, h,
                                                          grad_kernel)
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
                                              rho_a, rho_b, h, grad_kernel)
    (; epsilon, nu) = viscosity

    # TODO This is not correct for two different fluids. It should be `nu_a` and `nu_b`.
    mu_a = nu * rho_a
    mu_b = nu * rho_b

    return (mu_a + mu_b) / (rho_a * rho_b) * dot(pos_diff, grad_kernel) /
           (distance^2 + epsilon * h^2) * v_diff
end

# See, e.g.,
# Joseph J. Monaghan. "Smoothed Particle Hydrodynamics".
# In: Reports on Progress in Physics (2005), pages 1703-1759.
# [doi: 10.1088/0034-4885/68/8/r01](http://dx.doi.org/10.1088/0034-4885/68/8/R01)
function kinematic_viscosity(system, viscosity::ArtificialViscosityMonaghan)
    (; smoothing_length) = system
    (; alpha) = viscosity
    sound_speed = system_sound_speed(system)

    return alpha * smoothing_length * sound_speed / (2 * ndims(system) + 4)
end

@doc raw"""
    ViscosityAdami(; nu, epsilon=0.01)

Viscosity by Adami (Adami et al. 2012).
The viscous interaction is calculated with the shear force for incompressible flows given by
```math
f_{ab} = \sum_w \bar{\eta}_{ab} \left( V_a^2 + V_b^2 \right) \frac{v_{ab}}{||r_{ab}||^2+\epsilon h_{ab}^2}  \nabla W_{ab} \cdot r_{ab},
```
where ``r_{ab} = r_a - r_b`` is the difference of the coordinates of particles ``a`` and ``b``,
``v_{ab} = v_a - v_b`` is the difference of their velocities, ``h`` is the smoothing length and ``V`` is the particle volume.
The parameter ``\epsilon`` prevents singularities (see Ramachandran et al. 2019).
The inter-particle-averaged shear stress  is
```math
    \bar{\eta}_{ab} =\frac{2 \eta_a \eta_b}{\eta_a + \eta_b},
```
where ``\eta_a = \rho_a \nu_a`` with ``\nu`` as the kinematic viscosity.

# Keywords
- `nu`: Kinematic viscosity
- `epsilon=0.01`: Parameter to prevent singularities

## References
- S. Adami et al. "A generalized wall boundary condition for smoothed particle hydrodynamics".
  In: Journal of Computational Physics 231 (2012), pages 7057-7075.
  [doi: 10.1016/j.jcp.2012.05.005](http://dx.doi.org/10.1016/j.jcp.2012.05.005)
- P. Ramachandran et al. "Entropically damped artificial compressibility for SPH".
  In: Journal of Computers and Fluids 179 (2019), pages 579-594.
  [doi: 10.1016/j.compfluid.2018.11.023](https://doi.org/10.1016/j.compfluid.2018.11.023)
"""
struct ViscosityAdami{ELTYPE}
    nu::ELTYPE
    epsilon::ELTYPE

    function ViscosityAdami(; nu, epsilon=0.01)
        new{typeof(nu)}(nu, epsilon)
    end
end

@inline function (viscosity::ViscosityAdami)(particle_system, neighbor_system,
                                             v_particle_system, v_neighbor_system,
                                             particle, neighbor, pos_diff,
                                             distance, sound_speed, m_a, m_b,
                                             rho_a, rho_b, grad_kernel)
    (; epsilon, nu) = viscosity
    (; smoothing_length) = particle_system

    v_a = viscous_velocity(v_particle_system, particle_system, particle)
    v_b = viscous_velocity(v_neighbor_system, neighbor_system, neighbor)
    v_diff = v_a - v_b

    # TODO This is not correct for two different fluids. It should be `nu_a` and `nu_b`.
    eta_a = nu * rho_a
    eta_b = nu * rho_b

    eta_tilde = 2 * (eta_a * eta_b) / (eta_a + eta_b)

    # TODO For variable smoothing_length use average smoothing length
    tmp = eta_tilde / (distance^2 + epsilon * smoothing_length^2)

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

function kinematic_viscosity(system, viscosity::ViscosityAdami)
    return viscosity.nu
end

@inline viscous_velocity(v, system, particle) = current_velocity(v, system, particle)

struct NoViscosity end

@inline function (::NoViscosity)(particle_system, neighbor_system, v_particle_system,
                                 v_neighbor_system, particle, neighbor, pos_diff, distance,
                                 sound_speed, m_a, m_b, rho_mean)
    return SVector(ntuple(_ -> 0.0, Val(ndims(particle_system))))
end

@doc raw"""
    ArtificialViscosityMonaghan(; alpha, beta, epsilon=0.01)

# Keywords
- `alpha`: A value of `0.02` is usually used for most simulations. For a relation with the
           kinematic viscosity, see description below.
- `beta`: A value of `0.0` works well for simulations with shocks of moderate strength.
          In simulations where the Mach number can be very high, eg. astrophysical calculation,
          good results can be obtained by choosing a value of `beta=2` and `alpha=1`.
- `epsilon=0.01`: Parameter to prevent singularities.


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
\nu = \frac{\alpha h c }{\rho_{ab}}.
```

## References:
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

    function ArtificialViscosityMonaghan(; alpha, beta, epsilon=0.01)
        new{typeof(alpha)}(alpha, beta, epsilon)
    end
end

@inline function (viscosity::ArtificialViscosityMonaghan)(particle_system, neighbor_system,
                                                          v_particle_system,
                                                          v_neighbor_system,
                                                          particle, neighbor, pos_diff,
                                                          distance, sound_speed, m_a, m_b,
                                                          rho_mean)
    (; smoothing_length) = particle_system

    v_a = current_velocity(v_particle_system, particle_system, particle)
    v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)
    v_diff = v_a - v_b

    pi_ab = viscosity(sound_speed, v_diff, pos_diff, distance, rho_mean, smoothing_length)

    if pi_ab < eps()
        return SVector(ntuple(_ -> 0.0, Val(ndims(particle_system))))
    end

    return -m_b * pi_ab * smoothing_kernel_grad(particle_system, pos_diff, distance)
end

@inline function (viscosity::ArtificialViscosityMonaghan)(c, v_diff, pos_diff, distance,
                                                          rho_mean, h)
    (; alpha, beta, epsilon) = viscosity

    # v_ab ⋅ r_ab
    vr = dot(v_diff, pos_diff)

    # Monaghan 2005 p. 1741 (doi: 10.1088/0034-4885/68/8/r01):
    # "In the case of shock tube problems, it is usual to turn the viscosity on for
    # approaching particles and turn it off for receding particles. In this way, the
    # viscosity is used for shocks and not rarefactions."
    if vr < 0
        mu = h * vr / (distance^2 + epsilon * h^2)
        return -(alpha * c * mu + beta * mu^2) / rho_mean
    end

    return 0.0
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

## References:
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
                                             distance, sound_speed, m_a, m_b, rho_mean)
    (; epsilon, nu) = viscosity
    (; smoothing_length) = particle_system

    v_a = viscous_velocity(v_particle_system, particle_system, particle)
    v_b = viscous_velocity(v_neighbor_system, neighbor_system, neighbor)
    v_diff = v_a - v_b

    rho_a = particle_density(v_particle_system, particle_system, particle)
    rho_b = particle_density(v_neighbor_system, neighbor_system, neighbor)

    eta_a = nu * rho_a
    eta_b = nu * rho_b

    eta_tilde = 2 * (eta_a * eta_b) / (eta_a + eta_b)

    # TODO For variable smoothing_length use average smoothing length
    tmp = eta_tilde / (distance^2 + epsilon * smoothing_length^2)

    volume_a = m_a / rho_a
    volume_b = m_b / rho_b

    grad_kernel = smoothing_kernel_grad(particle_system, pos_diff, distance)

    visc = (volume_a^2 + volume_b^2) * dot(grad_kernel, pos_diff) * tmp / m_a

    return visc .* v_diff
end

@inline viscous_velocity(v, system, particle) = current_velocity(v, system, particle)

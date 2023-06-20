struct NoViscosity end

@inline function (::NoViscosity)(particle_system, neighbor_system, v_particle_system,
                                 v_neighbor_system, particle, neighbor, pos_diff, distance,
                                 sound_speed, m_a, m_b, rho_mean)
    return SVector(ntuple(_ -> 0.0, Val(ndims(particle_system))))
end

@doc raw"""
    ArtificialViscosityMonaghan(alpha, beta, epsilon=0.01)

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

TODO: Check the following statement, since in Monaghan 2005 p. 1741 (10.1088/0034-4885/68/8/r01) this was meant for "interstellar cloud collisions"
The choice of the parameters ``\alpha`` and ``\beta`` is not critical, but their values should usually be near
``\alpha = 1, \beta = 2`` (Monaghan 1992, p. 551).
The parameter ``\epsilon`` prevents singularities and is usually chosen as ``\epsilon = 0.01``.

Note that ``\alpha`` needs to adjusted for different resolutions to maintain a specific Reynolds Number.
To do so, Monaghan (Monaghan 2005) defined an equivalent effecive physical kinematic viscosity ``\nu`` by
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

    function ArtificialViscosityMonaghan(alpha, beta, epsilon=0.01)
        new{typeof(alpha)}(alpha, beta, epsilon)
    end
end

@inline function (viscosity::ArtificialViscosityMonaghan)(particle_system, neighbor_system,
                                                          v_particle_system,
                                                          v_neighbor_system,
                                                          particle, neighbor, pos_diff,
                                                          distance, sound_speed, m_a, m_b,
                                                          rho_mean)
    @unpack smoothing_length = particle_system

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
    @unpack alpha, beta, epsilon = viscosity

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

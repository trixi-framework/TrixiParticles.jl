struct NoViscosity end

function (::NoViscosity)(c, v_diff, pos_diff, distance, density_mean, h)
    return 0.0
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

The choice of the parameters ``\alpha`` and ``\beta`` is not critical, but their values should usually be near
``\alpha = 1, \beta = 2`` (Monaghan 1992, p. 551).
The parameter ``\epsilon`` prevents singularities and is usually chosen as ``\epsilon = 0.01``.

References:
- Joseph J. Monaghan. "Smoothed Particle Hydrodynamics".
  In: Annual Review of Astronomy and Astrophysics 30.1 (1992), pages 543-574.
  [doi: 10.1146/ANNUREV.AA.30.090192.002551](https://doi.org/10.1146/ANNUREV.AA.30.090192.002551)
- Joseph J. Monaghan. "On the Problem of Penetration in Particle Methods".
  In: Journal of Computational Physics 82.1, pages 1–15.
  [doi: 10.1016/0021-9991(89)90032-6](https://doi.org/10.1016/0021-9991(89)90032-6)
"""
struct ArtificialViscosityMonaghan{ELTYPE}
    alpha   ::ELTYPE
    beta    ::ELTYPE
    epsilon ::ELTYPE

    function ArtificialViscosityMonaghan(alpha, beta, epsilon=0.01)
        new{typeof(alpha)}(alpha, beta, epsilon)
    end
end

function (viscosity::ArtificialViscosityMonaghan)(c, v_diff, pos_diff, distance, density_mean, h)
    @unpack alpha, beta, epsilon = viscosity

    # v_ab ⋅ r_ab
    vr = sum(v_diff .* pos_diff)

    if vr < 0
        mu = h * vr / (distance^2 + epsilon * h^2)
        return (alpha * c * mu + beta * mu^2) / density_mean
    end

    return 0.0
end

@doc raw"""
    ViscosityClearyMonaghan(nu)
Viscosity by Cleary and Monaghan, given by
```math
\left(\nu \nabla^2 \vec{u} \right)_i= \sum_{j=1}^N \frac{8 (\nu_i + \nu_j)}{\rho_i + \rho_j}\frac{ \vec{r_{ij}} \cdot \vec{u_{ij}} }{r_{ij}^2 + \eta^2} \nabla_i W_{ij},
```
where ``\vec{u_{ij}} = \vec{u_{i}} - \vec{u_{j}} ``, ``\eta = 0.1 h`` is to keep the denominator nonzero and ``\nu`` is the kinematic viscosity.

References:
- P.W. Cleary, J.J. Monahgan "Conduction Modelling Using Smoothed Particle Hydrodynamics".
  In: Powder Technology 400 (1999).
  [doi: 10.1006/jcph.1998.6118](https://doi.org/10.1006/jcph.1998.6118)
  !!! note "TBD"

"""
struct ViscosityClearyMonaghan{ELTYPE}
    nu   ::ELTYPE

    function ViscosityClearyMonaghan(nu)
        new{typeof(nu)}(nu)
    end
end

function (viscosity::ViscosityClearyMonaghan)(c, v_diff, pos_diff, distance, density_particle, h)
    @unpack nu = viscosity
    eta = 0.1*h # to keep the denominator nonzero.
    vr = sum(pos_diff .* v_diff)

    return 8*nu*vr/((density_particle)*(distance^2+eta^2))

end
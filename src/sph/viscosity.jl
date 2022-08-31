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


!!! note "TBD"
    - RE> 1: Monaghan’s formulation
    - RE< 1 Morris’ formulation 
    References:
    - Alexiadis, Alessio. "The Discrete Multi-Hybrid System for the Simulation of Solid-Liquid Flows "
    doi: 10.1371/journal.pone.0124678
"""
struct ArtificialViscosityMonaghan{ELTYPE}
    alpha   ::ELTYPE
    beta    ::ELTYPE
    epsilon ::ELTYPE

    function ArtificialViscosityMonaghan(alpha, beta, epsilon=0.01)
        new{typeof(alpha)}(alpha, beta, epsilon)
    end
end

function (viscosity::ArtificialViscosityMonaghan)(c, v_diff, pos_diff, distance, density_particle, density_neighbor, h)
    @unpack alpha, beta, epsilon = viscosity
    density_mean = (density_particle + density_neighbor) / 2

    # v_ab ⋅ r_ab
    vr = sum(v_diff .* pos_diff)

    if vr < 0
        mu = h * vr / (distance^2 + epsilon * h^2)
        return -(alpha * c * mu + beta * mu^2) / density_mean
    end

    return 0.0
end

@doc raw"""
    ViscosityMorris(mu)
References:
- Lu Liu. "DEM–SPH coupling method for the interaction between irregularly shaped granular materials and ﬂuids".
  In: Powder Technology 400 (2022).
  [doi: 10.1016/j.powtec.2022.117249](https://doi.org/10.1016/j.powtec.2022.117249)
  !!! note "TBD"

"""
struct ViscosityMorris{ELTYPE}
    nu   ::ELTYPE

    function ViscosityMorris(nu)
        new{typeof(nu)}(nu)
    end
end

function (viscosity::ViscosityMorris)(c, v_diff, pos_diff, distance, density_particle, density_neighbor, h)
    @unpack nu = viscosity

    vr = sum(v_diff .* pos_diff)

end
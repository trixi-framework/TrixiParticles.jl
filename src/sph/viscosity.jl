struct NoViscosity end

function (::NoViscosity)(c, v_diff, pos_diff, distance, rho_mean, h)
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

function (viscosity::ArtificialViscosityMonaghan)(c, v_diff, pos_diff, distance,
                                                  rho_mean, h)
    @unpack alpha, beta, epsilon = viscosity

    # v_ab ⋅ r_ab
    vr = sum(v_diff .* pos_diff)

    # Monaghan 2005 p. 1741 (doi: 10.1088/0034-4885/68/8/r01):
    # "In the case of shock tube problems, it is usual to turn the viscosity on for
    # approaching  particles and turn it off for receding particles. In this way, the
    # viscosity is used for shocks and not rarefactions."
    if vr < 0
        mu = h * vr / (distance^2 + epsilon * h^2)
        return -(alpha * c * mu + beta * mu^2) / rho_mean
    end

    return 0.0
end

@doc raw"""
    ViscousInteractionAdami(eta, coords)

# Arguments
- `eta`: Inter-particle-averaged shear stress
- `coords`: Particle coordinates

Shear force for the interaction of dummy particles (see [`BoundaryModelDummyParticles`](@ref)) and fluid particles.
Since the [`ArtificialViscosityMonaghan`](@ref) is only applicable for the [`BoundaryModelMonaghanKajtar`](@ref),
Adami (Adami et al 2012) imposes a no-slip boundary condition by extrapolating the smoothed
velocity field of the fluid to the dummy particle position.
The viscous interaction is then calculated with the shear force for incompressible flows given by
```math
f_{fw} = \sum_w \bar{\eta}_{fw} \left( V_f^2 + V_w^2 \right) \frac{v_{fw}}{||r_{fw}||} \frac{\partial W}{\partial ||r_{fw}||},
```
where the subindices ``f`` and ``w`` denote fluid and boundary particles respectively,
``\bar{\eta}_{fw}`` is the inter-particle-averaged shear stress and ``V`` is the particle volume.

The velocity of particle ``w`` is calculated by the prescribed boundary particle velocity ``v_a`` and the exptrapolated velocity:
```math
v_w = 2 v_a - \frac{\sum_b v_b W_{ab}}{\sum_b W_{ab}},
```
where the sum is over all fluid particles.

## References:
- S. Adami et al. "A generalized wall boundary condition for smoothed particle hydrodynamics".
  In: Journal of Computational Physics 231 (2012), pages 7057-7075.
  [doi: 10.1016/j.jcp.2012.05.005](http://dx.doi.org/10.1016/j.jcp.2012.05.005)
"""
struct ViscousInteractionAdami{ELTYPE}
    eta::ELTYPE
    velocities::Array{ELTYPE, 2}

    function ViscousInteractionAdami(eta, coords)
        velocities = zeros(eltype(coords), size(coords))

        new{typeof(eta)}(eta, velocities)
    end
end

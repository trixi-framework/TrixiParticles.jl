# [Open Boundary System](@id open_boundary)
The difficulty in non-reflecting boundary conditions, also called open boundaries, is to determine
the appropriate boundary values of the exact characteristics of the Euler equations.
Giles (1990) derived three characteristic variables which are constant along curves in ``x``-``t`` plane
defined by
```math
\underbrace{\frac{\partial x}{\partial t} = v + c_s, \quad \frac{\partial x}{\partial t} = v}_{\text{downstream-running}},
        \quad
        \text{and} \underbrace{\frac{\partial x}{\partial t} = v - c_s}_{\text{upstream-running (for subsonic
                flow)}}
```
and can be interpreted as the trajectories of the sound waves and material particles that carry constant values of the
characteristic variables.

The characteristic variables based on a linearized set of governing equations are given as
```math
J_1 = -c_s^2 (\rho - \rho_{\text{ref}}) + (p - p_{\text{ref}})
```
```math
J_2 = \rho c_s (v - v_{\text{ref}}) + (p - p_{\text{ref}})
```
```math
J_3 = - \rho c_s (v - v_{\text{ref}}) + (p - p_{\text{ref}})
```
where the subscript "ref" denotes the reference flow near the boundaries, which can be prescribed.

Specifying the reference variables is **not** equivalent to prescription of ``\rho``, ``v`` and ``p``
directly, since the perturbation from the reference flow is allowed.

Lastiwka et al (2009) applied the method of characteristic to SPH and determine the number of variables that should be
**prescribed** at the boundary and the number which should be **propagated** from the fluid domain to the boundary:

Flow enters the domain through an
- **inflow** boundary:
    - Prescribe *downstream*-running characteristics ``J_1`` and ``J_2``
    - Transmit ``J_3`` from the fluid domain (allow ``J_3`` to propagate upstream to the boundary).

- **outflow** boundary:
    - Prescribe *upstream*-running characteristic ``J_3``
    - Transmit ``J_1`` and ``J_2`` from the fluid domain.

Prescribing is done by simply setting the characteristics to zero. To transmit the characteristics from the fluid
domain, or in other words, to carry the information of the fluid to the boundaries, Negi (2020) use a Shepard Interpolation
```math
f_i = \frac{\sum_j^N f_j W_{ij}}{\sum_j^N W_{ij}},
```
where the ``i``th particle is a boundary particle, ``f`` is either  ``J_1``, ``J_2`` or ``J_3`` and ``N`` is the set of
neighboring fluid particles.

To express pressure ``p``, density ``\rho`` and velocity ``v`` as functions of the characteristic variables, the system of equations
from the characteristic variables is inverted and gives
```math
 \rho - \rho_{\text{ref}} = \frac{1}{c_s^2} \left( -J_1 + \frac{1}{2} J_2 + \frac{1}{2} J_3 \right),
```
```math
u - u_{\text{ref}}= \frac{1}{2\rho c_s} \left( J_2 - J_3 \right),
```
```math
p - p_{\text{ref}} = \frac{1}{2} \left( J_2 + J_3 \right).
```
Thus, determined ``J_1``, ``J_2`` and ``J_3``, we can easily solve for the actual variables for each particle.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "boundary", "open_boundary", "system.jl")]
```

## References
- M. B. Giles "Nonreflecting boundary conditions for Euler equation calculations".
  In: AIAA Journal , Vol. 28, No. 12 pages 2050--2058
  [doi: 10.2514/3.10521](https://doi.org/10.2514/3.10521)
- M. Lastiwka, M. Basa, N. J. Quinlan.
  "Permeable and non-reflecting boundary conditions in SPH".
  In: International Journal for Numerical Methods in Fluids 61, (2009), pages 709--724.
  [doi: 10.1002/fld.1971](https://doi.org/10.1002/fld.1971)
- P. Negi, P. Ramachandran, A. Haftu.
  "An improved non-reflecting outlet boundary condition for weakly-compressible SPH".
  In: Computer Methods in Applied Mechanics and Engineering 367, (2020), pages 113119.
  [doi: 10.1016/j.cma.2020.113119](https://doi.org/10.1016/j.cma.2020.113119)

@doc raw"""
    PenaltyForceGanzenmueller(; alpha=0.1)

In FEM there are hour-glassing correction techniques for underintegrated (reduced number of integration points)
finite elements. This is, the element deforms without an associated increase of energy.
There is an analogy to SPH in the sense that the SPH approximation of the deformation gradient ``\bm{J}``
is not unique with respect to the positions of the integration points.
In order to circumvent this, (Georg C. Ganzenmüller 2015) introduced a so-called hourglass correction force or penalty force ``f^{HG}``,
which is given by
```math
\bm{f}_a^{HG} = \frac{1}{2} \alpha \sum_b \frac{V_{0a} V_{0b} W_{0ab}}{X_{ab}^2}
                \left( E \delta_{ab}^a + E \delta_{ba}^b \right) \frac{\bm{x}_{ab}}{x_{ab}}
```
where ``V_{0a}`` and ``V_{0b}`` is the unit voume of particle ``a`` and ``b`` respectively and
``W_{0ab}`` is a smoothing kernel. The subindex ``0`` indicates the reference configuration.
``X_{ab}`` is the distance of particle ``a`` and ``b`` in the reference configuration and ``x_{ab}`` is
the distance in the current configuration.

This correction force is based on the potential energy density of a Hookean material.
Thus, ``E`` is the Young's modulus and ``\alpha`` is a dimensionless coefficient that controls
the amplitude of hourglass correction.
The separation vector $\delta_{ab}^a$ indicates the change of distance which the particle separation should attain
in order to minimize the error and is given by
```math
\delta_{ab}^a = \frac{\bm{\epsilon}_{ab}^a\bm{x_{ab}}}{x_{ab}}
```
where the error vector is defined as
```math
\bm{\epsilon}_{ab}^a = \bm{J}_a \bm{X}_{ab} - \bm{x}_{ab} .
```

References:
- Georg C. Ganzenmüller.
  "An hourglass control algorithm for Lagrangian Smooth Particle Hydrodynamics".
  In: Computer Methods in Applied Mechanics and Engineering 286 (2015).
  [doi: 10.1016/j.cma.2014.12.005](https://doi.org/10.1016/j.cma.2014.12.005)
"""
struct PenaltyForceGanzenmueller{ELTYPE}
    alpha   :: ELTYPE
    function PenaltyForceGanzenmueller(; alpha=1.0)
        new{typeof(alpha)}(alpha)
    end
end

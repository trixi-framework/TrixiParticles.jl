@doc raw"""
    PenaltyForceGanzenmueller(; alpha=0.1)

Penalty force to ensure regular particle positions under large deformations.

In FEM, underintegrated elements can deform without an associated increase of energy.
This is caused by the stiffness matrix having zero eigenvalues (so-called hourglass modes).
The name "hourglass modes" comes from the fact that elements can deform into an hourglass shape.

Similar effects can occur in SPH as well.
Particles can change positions without changing the SPH approximation of the deformation gradient $\bm{J}$,
thus, without causing an increase of energy.
To ensure regular particle positions, we can apply similar correction forces as are used in FEM.

(Ganzenmüller, 2015) introduced a so-called hourglass correction force or penalty force $f^{PF}$,
which is given by
```math
\bm{f}_a^{PF} = \frac{1}{2} \alpha \sum_b \frac{m_{0a} m_{0b} W_{0ab}}{\rho_{0a}\rho_{0b} |\bm{X}_{ab}|^2}
                \left( E \delta_{ab}^a + E \delta_{ba}^b \right) \frac{\bm{x}_{ab}}{|\bm{x}_{ab}|}
```
The subscripts $a$ and $b$ denote quantities of particle $a$ and $b$, respectively.
The zero subscript on quantities denotes that the quantity is to be measured in the initial configuration.
The difference in the initial coordinates is denoted by $\bm{X}_{ab} = \bm{X}_a - \bm{X}_b$,
the difference in the current coordinates is denoted by $\bm{x}_{ab} = \bm{x}_a - \bm{x}_b$.
Note that (Ganzenmüller, 2015) has a flipped sign here because they define $\bm{x}_{ab}$ the other way around.

This correction force is based on the potential energy density of a Hookean material.
Thus, $E$ is the Young's modulus and $\alpha$ is a dimensionless coefficient that controls
the amplitude of hourglass correction.
The separation vector $\delta_{ab}^a$ indicates the change of distance which the particle separation should attain
in order to minimize the error and is given by
```math
    \delta_{ab}^a = \frac{\bm{\epsilon}_{ab}^a \cdot \bm{x_{ab}}}{|\bm{x}_{ab}|},
```
where the error vector is defined as
```math
    \bm{\epsilon}_{ab}^a = \bm{J}_a \bm{X}_{ab} - \bm{x}_{ab}.
```


References:
- Georg C. Ganzenmüller.
  "An hourglass control algorithm for Lagrangian Smooth Particle Hydrodynamics".
  In: Computer Methods in Applied Mechanics and Engineering 286 (2015).
  [doi: 10.1016/j.cma.2014.12.005](https://doi.org/10.1016/j.cma.2014.12.005)
"""
struct PenaltyForceGanzenmueller{ELTYPE}
    alpha::ELTYPE
    function PenaltyForceGanzenmueller(; alpha=0.1)
        new{typeof(alpha)}(alpha)
    end
end

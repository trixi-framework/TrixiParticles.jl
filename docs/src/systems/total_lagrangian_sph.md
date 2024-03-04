# [Total Lagrangian SPH](@id tlsph)

A Total Lagrangian framework is used wherein the governing equations are formulated such that
all relevant quantities and operators are measured with respect to the
initial configuration (O’Connor & Rogers 2021, Belytschko et al. 2000).

The governing equations with respect to the initial configuration are given by:
```math
\frac{\mathrm{D}\bm{v}}{\mathrm{D}t} = \frac{1}{\rho_0} \nabla_0 \cdot \bm{P} + \bm{g},
```
where the zero subscript denotes a derivative with respect to the initial configuration
and $\bm{P}$ is the first Piola-Kirchhoff (PK1) stress tensor.

The discretized version of this equation is given by O’Connor & Rogers (2021):
```math
\frac{\mathrm{d}\bm{v}_a}{\mathrm{d}t} = \sum_b m_{0b}
    \left( \frac{\bm{P}_a \bm{L}_{0a}}{\rho_{0a}^2} + \frac{\bm{P}_b \bm{L}_{0b}}{\rho_{0b}^2} \right)
    \nabla_{0a} W(\bm{X}_{ab}) + \frac{\bm{f}_a^{PF}}{m_{0a}} + \bm{g},
```
with the correction matrix (see also [`GradientCorrection`](@ref))
```math
\bm{L}_{0a} := \left( -\sum_{b} \frac{m_{0b}}{\rho_{0b}} \nabla_{0a} W(\bm{X}_{ab}) \bm{X}_{ab}^T \right)^{-1} \in \R^{d \times d}.
```
The subscripts $a$ and $b$ denote quantities of particle $a$ and $b$, respectively.
The zero subscript on quantities denotes that the quantity is to be measured in the initial configuration.
The difference in the initial coordinates is denoted by $\bm{X}_{ab} = \bm{X}_a - \bm{X}_b$,
the difference in the current coordinates is denoted by $\bm{x}_{ab} = \bm{x}_a - \bm{x}_b$.

For the computation of the PK1 stress tensor, the deformation gradient $\bm{F}$ is computed per particle as
```math
\bm{F}_a = \sum_b \frac{m_{0b}}{\rho_{0b}} \bm{x}_{ba} (\bm{L}_{0a}\nabla_{0a} W(\bm{X}_{ab}))^T \\
    \qquad  = -\left(\sum_b \frac{m_{0b}}{\rho_{0b}} \bm{x}_{ab} (\nabla_{0a} W(\bm{X}_{ab}))^T \right) \bm{L}_{0a}^T
```
with $1 \leq i,j \leq d$.
From the deformation gradient, the Green-Lagrange strain
```math
\bm{E} = \frac{1}{2}(\bm{F}^T\bm{F} - \bm{I})
```
and the second Piola-Kirchhoff stress tensor
```math
\bm{S} = \lambda \operatorname{tr}(\bm{E}) \bm{I} + 2\mu \bm{E}
```
are computed to obtain the PK1 stress tensor as
```math
\bm{P} = \bm{F}\bm{S}.
```

Here,
```math
\mu = \frac{E}{2(1 + \nu)}
```
and
```math
\lambda = \frac{E\nu}{(1 + \nu)(1 - 2\nu)}
```
are the Lamé coefficients, where $E$ is the Young's modulus and $\nu$ is the Poisson ratio.

The term $\bm{f}_a^{PF}$ is an optional penalty force. See e.g. [`PenaltyForceGanzenmueller`](@ref).

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "solid", "total_lagrangian_sph", "system.jl")]
```

### References
- Joseph O’Connor, Benedict D. Rogers.
  "A fluid-structure interaction model for free-surface flows and flexible structures using
  smoothed particle hydrodynamics on a GPU".
  In: Journal of Fluids and Structures 104 (2021).
  [doi: 10.1016/J.JFLUIDSTRUCTS.2021.103312](https://doi.org/10.1016/J.JFLUIDSTRUCTS.2021.103312)
- Ted Belytschko, Yong Guo, Wing Kam Liu, Shao Ping Xiao.
  "A unified stability analysis of meshless particle methods".
  In: International Journal for Numerical Methods in Engineering 48 (2000), pages 1359–1400.
  [doi: 10.1002/1097-0207](https://doi.org/10.1002/1097-0207)

## Penalty Force

In FEM, underintegrated elements can deform without an associated increase of energy.
This is caused by the stiffness matrix having zero eigenvalues (so-called hourglass modes).
The name "hourglass modes" comes from the fact that elements can deform into an hourglass shape.

Similar effects can occur in SPH as well.
Particles can change positions without changing the SPH approximation of the deformation gradient $\bm{F}$,
thus, without causing an increase of energy.
To ensure regular particle positions, we can apply similar correction forces as are used in FEM.

Ganzenmüller (2015) introduced a so-called hourglass correction force or penalty force $f^{PF}$,
which is given by
```math
\bm{f}_a^{PF} = \frac{1}{2} \alpha \sum_b \frac{m_{0a} m_{0b} W_{0ab}}{\rho_{0a}\rho_{0b} |\bm{X}_{ab}|^2}
                \left( E \delta_{ab}^a + E \delta_{ba}^b \right) \frac{\bm{x}_{ab}}{|\bm{x}_{ab}|}
```
The subscripts $a$ and $b$ denote quantities of particle $a$ and $b$, respectively.
The zero subscript on quantities denotes that the quantity is to be measured in the initial configuration.
The difference in the initial coordinates is denoted by $\bm{X}_{ab} = \bm{X}_a - \bm{X}_b$,
the difference in the current coordinates is denoted by $\bm{x}_{ab} = \bm{x}_a - \bm{x}_b$.
Note that Ganzenmüller (2015) has a flipped sign here because they define $\bm{x}_{ab}$ the other way around.

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
    \bm{\epsilon}_{ab}^a = \bm{F}_a \bm{X}_{ab} - \bm{x}_{ab}.
```

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "solid", "total_lagrangian_sph", "penalty_force.jl")]
```

### References
- Georg C. Ganzenmüller.
  "An hourglass control algorithm for Lagrangian Smooth Particle Hydrodynamics".
  In: Computer Methods in Applied Mechanics and Engineering 286 (2015).
  [doi: 10.1016/j.cma.2014.12.005](https://doi.org/10.1016/j.cma.2014.12.005)

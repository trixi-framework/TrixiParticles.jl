# [Total Lagrangian SPH](@id tlsph)

A Total Lagrangian framework is used wherein the governing equations are formulated such that
all relevant quantities and operators are measured with respect to the
initial configuration ([O’Connor & Rogers, 2021](@cite OConnor2021), [Belytschko et al., 2000](@cite Belytschko2000)).

The governing equations with respect to the initial configuration are given by:
```math
\frac{\mathrm{D}\bm{v}}{\mathrm{D}t} = \frac{1}{\rho_0} \nabla_0 \cdot \bm{P} + \bm{g},
```
where the zero subscript denotes a derivative with respect to the initial configuration
and $\bm{P}$ is the first Piola-Kirchhoff (PK1) stress tensor.

The discretized version of this equation is given by [O’Connor & Rogers (2021)](@cite OConnor2021):
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
Pages = [joinpath("schemes", "structure", "total_lagrangian_sph", "system.jl")]
```

## Penalty Force

In FEM, underintegrated elements can deform without an associated increase of energy.
This is caused by the stiffness matrix having zero eigenvalues (so-called hourglass modes).
The name "hourglass modes" comes from the fact that elements can deform into an hourglass shape.

Similar effects can occur in SPH as well.
Particles can change positions without changing the SPH approximation of the deformation gradient $\bm{F}$,
thus, without causing an increase of energy.
To ensure regular particle positions, we can apply similar correction forces as are used in FEM.

[Ganzenmüller (2015)](@cite Ganzenmueller2015) introduced a so-called hourglass correction force or penalty force $f^{PF}$,
which is given by
```math
\bm{f}_a^{PF} = \frac{1}{2} \alpha \sum_b \frac{m_{0a} m_{0b} W_{0ab}}{\rho_{0a}\rho_{0b} |\bm{X}_{ab}|^2}
                \left( E \delta_{ab}^a + E \delta_{ba}^b \right) \frac{\bm{x}_{ab}}{|\bm{x}_{ab}|}
```
The subscripts $a$ and $b$ denote quantities of particle $a$ and $b$, respectively.
The zero subscript on quantities denotes that the quantity is to be measured in the initial configuration.
The difference in the initial coordinates is denoted by $\bm{X}_{ab} = \bm{X}_a - \bm{X}_b$,
the difference in the current coordinates is denoted by $\bm{x}_{ab} = \bm{x}_a - \bm{x}_b$.
Note that [Ganzenmüller (2015)](@cite Ganzenmueller2015) has a flipped sign here because they define $\bm{x}_{ab}$ the other way around.

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
Pages = [joinpath("schemes", "structure", "total_lagrangian_sph", "penalty_force.jl")]
```

## Viscosity

Another technique that is used to correct the hourglass instability is artificial viscosity.
Hereby, a viscosity term designed for fluids (see [Viscosity](@ref viscosity_sph)) is applied.
First, the force ``f_{ab}^{\text{fluid}}`` exerted by particle ``b`` on particle ``a``
due to artificial viscosity is computed as if both particles were fluid particles
(see [Viscosity](@ref viscosity_sph) for the relevant equations).
Then, according to [Lin et al. (2015)](@cite Lin2015), this force can be applied to TLSPH
with the following conversion:
```math
f_{ab}^{\text{AV}} = \det(F_a) F_a^{-1} f_{ab}^{\text{fluid}},
```
where ``F_a`` is the deformation gradient at particle ``a``.

We found that artificial viscosity is not effective at correcting the incorrect
particle positions due to hourglass modes.
It does however prevent particles from oscillating between different incorrect positions,
which develops into an instability if uncorrected.
We still recommend penalty force over artificial viscosity to correct hourglass modes
as penalty force is specifically designed to correct the incorrect particle positions.

In some FSI simulations, notably when very thin structures or structures with low material
density are present, instabilities in the fluid can be induced by the structure.
In these cases, artificial viscosity is effective at stabilizing the fluid close to the
structure, and we recommend using it in combination with penalty force to both
prevent hourglass modes and stabilize the fluid close to the fluid-structure interface.

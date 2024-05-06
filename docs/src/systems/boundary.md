# Boundary System

```@docs
    BoundarySPHSystem
```

```@docs
    BoundaryDEMSystem
```

```@docs
    BoundaryMovement
```


# [Boundary Models](@id boundary_models)

## Dummy Particles

Boundaries modeled as dummy particles, which are treated like fluid particles,
but their positions and velocities are not evolved in time. Since the force towards the fluid
should not change with the material density when used with a [`TotalLagrangianSPHSystem`](@ref), the
dummy particles need to have a mass corresponding to the fluid's rest density, which we call
"hydrodynamic mass", as opposed to mass corresponding to the material density of a
[`TotalLagrangianSPHSystem`](@ref).

Here, `initial_density` and `hydrodynamic_mass` are vectors that contains the initial density
and the hydrodynamic mass respectively for each boundary particle.
Note that when used with [`SummationDensity`](@ref) (see below), this is only used to determine
the element type and the number of boundary particles.

To establish a relationship between density and pressure, a `state_equation` has to be passed,
which should be the same as for the adjacent fluid systems.
To sum over neighboring particles, a `smoothing_kernel` and `smoothing_length` needs to be passed.
This should be the same as for the adjacent fluid system with the largest smoothing length.

In the literature, this kind of boundary particles is referred to as
"dummy particles" (Adami et al., 2012 and Valizadeh & Monaghan, 2015),
"frozen fluid particles" (Akinci et al., 2012) or "dynamic boundaries (Crespo et al., 2007).
The key detail of this boundary condition and the only difference between the boundary models
in these references is the way the density and pressure of boundary particles is computed.

Since boundary particles are treated like fluid particles, the force
on fluid particle ``a`` due to boundary particle ``b`` is given by
```math
f_{ab} = m_a m_b \left( \frac{p_a}{\rho_a^2} + \frac{p_b}{\rho_b^2} \right) \nabla_{r_a} W(\Vert r_a - r_b \Vert, h).
```
The quantities to be defined here are the density ``\rho_b`` and pressure ``p_b``
of the boundary particle ``b``.

```@docs
    BoundaryModelDummyParticles
```


### Hydrodynamic density of dummy particles

We provide five options to compute the boundary density and pressure, determined by the `density_calculator`:
1. (Recommended) With [`AdamiPressureExtrapolation`](@ref), the pressure is extrapolated from the pressure of the
   fluid according to (Adami et al., 2012), and the density is obtained by applying the inverse of the state equation.
   This option usually yields the best results of the options listed here.
2. With [`SummationDensity`](@ref), the density is calculated by summation over the neighboring particles,
   and the pressure is computed from the density with the state equation.
3. With [`ContinuityDensity`](@ref), the density is integrated from the continuity equation,
   and the pressure is computed from the density with the state equation.
   Note that this causes a gap between fluid and boundary where the boundary is initialized
   without any contact to the fluid. This is due to overestimation of the boundary density
   as soon as the fluid comes in contact with boundary particles that initially did not have
   contact to the fluid.
   Therefore, in dam break simulations, there is a visible "step", even though the boundary is supposed to be flat.
   See also [dual.sphysics.org/faq/#Q_13](https://dual.sphysics.org/faq/#Q_13).
4. With [`PressureZeroing`](@ref), the density is set to the reference density and the pressure
   is computed from the density with the state equation.
   This option is not recommended. The other options yield significantly better results.
5. With [`PressureMirroring`](@ref), the density is set to the reference density. The pressure
   is not used. Instead, the fluid pressure is mirrored as boundary pressure in the
   momentum equation.
   This option is not recommended due to stability issues. See [`PressureMirroring`](@ref)
   for more details.

#### 1. [`AdamiPressureExtrapolation`](@ref)

The pressure of the boundary particles is obtained by extrapolating the pressure of the fluid
according to (Adami et al., 2012).
The pressure of a boundary particle ``b`` is given by
```math
p_b = \frac{\sum_f (p_f + \rho_f (\bm{g} - \bm{a}_b) \cdot \bm{r}_{bf}) W(\Vert r_{bf} \Vert, h)}{\sum_f W(\Vert r_{bf} \Vert, h)},
```
where the sum is over all fluid particles, ``\rho_f`` and ``p_f`` denote the density and pressure of fluid particle ``f``, respectively,
``r_{bf} = r_b - r_f`` denotes the difference of the coordinates of particles ``b`` and ``f``,
``\bm{g}`` denotes the gravitational acceleration acting on the fluid, and ``\bm{a}_b`` denotes the acceleration of the boundary particle ``b``.
```@docs
    AdamiPressureExtrapolation
```

#### 4. [`PressureZeroing`](@ref)

This is the simplest way to implement dummy boundary particles.
The density of each particle is set to the reference density and the pressure to the
reference pressure (the corresponding pressure to the reference density by the state equation).
```@docs
    PressureZeroing
```

#### 5. [`PressureMirroring`](@ref)

Instead of calculating density and pressure for each boundary particle, we modify the
momentum equation,
```math
\frac{\mathrm{d}v_a}{\mathrm{d}t} = -\sum_b m_b \left( \frac{p_a}{\rho_a^2} + \frac{p_b}{\rho_b^2} \right) \nabla_a W_{ab}
```
to replace the unknown density $\rho_b$ if $b$ is a boundary particle by the reference density
and the unknown pressure $p_b$ if $b$ is a boundary particle by the pressure $p_a$ of the
interacting fluid particle.
The momentum equation therefore becomes
```math
\frac{\mathrm{d}v_a}{\mathrm{d}t} = -\sum_f m_f \left( \frac{p_a}{\rho_a^2} + \frac{p_f}{\rho_f^2} \right) \nabla_a W_{af}
-\sum_b m_b \left( \frac{p_a}{\rho_a^2} + \frac{p_a}{\rho_0^2} \right) \nabla_a W_{ab},
```
where the first sum is over all fluid particles and the second over all boundary particles.

This approach was first mentioned by Akinci et al. (2012) and written down in this form
by Band et al. (2018).
```@docs
    PressureMirroring
```

### No-slip conditions

For the interaction of dummy particles and fluid particles, Adami et al. (2012)
impose a no-slip boundary condition by assigning a wall velocity ``v_w`` to the dummy particle.

The wall velocity of particle ``a`` is calculated from the prescribed boundary particle
velocity ``v_a`` and the smoothed velocity field
```math
v_w = 2 v_a - \frac{\sum_b v_b W_{ab}}{\sum_b W_{ab}},
```
where the sum is over all fluid particles.

By choosing the viscosity model [`ViscosityAdami`](@ref) for `viscosity`, a no-slip
condition is imposed. It is recommended to choose `nu` in the order of either the kinematic
viscosity parameter of the adjacent fluid or the equivalent from the artificial parameter
`alpha` of the adjacent fluid (``\nu = \frac{\alpha h c }{2d + 4}``). When omitting the
viscous interaction (default `viscosity=nothing`), a free-slip wall boundary
condition is applied.

!!! warning
    The viscosity model [`ArtificialViscosityMonaghan`](@ref) for [`BoundaryModelDummyParticles`](@ref)
    has not been verified yet.

### References
- S. Adami, X. Y. Hu, N. A. Adams.
  "A generalized wall boundary condition for smoothed particle hydrodynamics".
  In: Journal of Computational Physics 231, 21 (2012), pages 7057–7075.
  [doi: 10.1016/J.JCP.2012.05.005](https://doi.org/10.1016/J.JCP.2012.05.005)
- Alireza Valizadeh, Joseph J. Monaghan.
  "A study of solid wall models for weakly compressible SPH".
  In: Journal of Computational Physics 300 (2015), pages 5–19.
  [doi: 10.1016/J.JCP.2015.07.033](https://doi.org/10.1016/J.JCP.2015.07.033)
- Nadir Akinci, Markus Ihmsen, Gizem Akinci, Barbara Solenthaler, Matthias Teschner.
  "Versatile rigid-fluid coupling for incompressible SPH".
  ACM Transactions on Graphics 31, 4 (2012), pages 1–8.
  [doi: 10.1145/2185520.2185558](https://doi.org/10.1145/2185520.2185558)
- A. J. C. Crespo, M. Gómez-Gesteira, R. A. Dalrymple.
  "Boundary conditions generated by dynamic particles in SPH methods"
  In: Computers, Materials and Continua 5 (2007), pages 173-184.
  [doi: 10.3970/cmc.2007.005.173](https://doi.org/10.3970/cmc.2007.005.173)
- Stefan Band, Christoph Gissler, Andreas Peer, and Matthias Teschner.
  "MLS Pressure Boundaries for Divergence-Free and Viscous SPH Fluids."
  In: Computers & Graphics 76 (2018), pages 37–46.
  [doi: 10.1016/j.cag.2018.08.001](https://doi.org/10.1016/j.cag.2018.08.001)

## Repulsive Particles

Boundaries modeled as boundary particles which exert forces on the fluid particles (Monaghan, Kajtar, 2009).
The force on fluid particle ``a`` due to boundary particle ``b`` is given by
```math
f_{ab} = m_a \left(\tilde{f}_{ab} - m_b \Pi_{ab} \nabla_{r_a} W(\Vert r_a - r_b \Vert, h)\right)
```
with
```math
\tilde{f}_{ab} = \frac{K}{\beta^{n-1}} \frac{r_{ab}}{\Vert r_{ab} \Vert (\Vert r_{ab} \Vert - d)} \Phi(\Vert r_{ab} \Vert, h)
\frac{2 m_b}{m_a + m_b},
```
where ``m_a`` and ``m_b`` are the masses of fluid particle ``a`` and boundary particle ``b``
respectively, ``r_{ab} = r_a - r_b`` is the difference of the coordinates of particles
``a`` and ``b``, ``d`` denotes the boundary particle spacing and ``n`` denotes the number of
dimensions (see (Monaghan, Kajtar, 2009, Equation (3.1)) and (Valizadeh, Monaghan, 2015)).
Note that the repulsive acceleration $\tilde{f}_{ab}$ does not depend on the masses of
the boundary particles.
Here, ``\Phi`` denotes the 1D Wendland C4 kernel, normalized to ``1.77`` for ``q=0``
(Monaghan, Kajtar, 2009, Section 4), with ``\Phi(r, h) = w(r/h)`` and
```math
w(q) =
\begin{cases}
  (1.77/32) (1 + (5/2)q + 2q^2)(2 - q)^5  & \text{if } 0 \leq q < 2 \\
  0                                       & \text{if } q \geq 2.
\end{cases}
```

The boundary particles are assumed to have uniform spacing by the factor ``\beta`` smaller
than the expected fluid particle spacing.
For example, if the fluid particles have an expected spacing of ``0.3`` and the boundary particles
have a uniform spacing of ``0.1``, then this parameter should be set to ``\beta = 3``.
According to (Monaghan, Kajtar, 2009), a value of ``\beta = 3`` for the Wendland C4 that
we use here is reasonable for most computing purposes.

The parameter ``K`` is used to scale the force exerted by the boundary particles.
In (Monaghan, Kajtar, 2009), a value of ``gD`` is used for static tank simulations,
where ``g`` is the gravitational acceleration and ``D`` is the depth of the fluid.

The viscosity ``\Pi_{ab}`` is calculated according to the viscosity used in the
simulation, where the density of the boundary particle if needed is assumed to be
identical to the density of the fluid particle.

### No-slip condition

By choosing the viscosity model [`ArtificialViscosityMonaghan`](@ref) for `viscosity`,
a no-slip condition is imposed. When omitting the viscous interaction
(default `viscosity=nothing`), a free-slip wall boundary condition is applied.

!!! warning
    The no-slip conditions for `BoundaryModelMonaghanKajtar` have not been verified yet.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "boundary", "monaghan_kajtar", "monaghan_kajtar.jl")]
```

### References
- Joseph J. Monaghan, Jules B. Kajtar. "SPH particle boundary forces for arbitrary boundaries".
  In: Computer Physics Communications 180.10 (2009), pages 1811–1820.
  [doi: 10.1016/j.cpc.2009.05.008](https://doi.org/10.1016/j.cpc.2009.05.008)
- Alireza Valizadeh, Joseph J. Monaghan. "A study of solid wall models for weakly compressible SPH."
  In: Journal of Computational Physics 300 (2015), pages 5–19.
  [doi: 10.1016/J.JCP.2015.07.033](https://doi.org/10.1016/J.JCP.2015.07.033)

# Boundary System

```@docs
    WallBoundarySystem
```

```@docs
    BoundaryDEMSystem
```

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "boundary", "prescribed_motion.jl")]
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
"dummy particles" ([Adami et al., 2012](@cite Adami2012) and [Valizadeh & Monaghan, 2015](@cite Valizadeh2015)),
"frozen fluid particles" ([Akinci et al., 2012](@cite Akinci2012)) or "dynamic boundaries [Crespo et al., 2007](@cite Crespo2007).
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

We provide six options to compute the boundary density and pressure, determined by the `density_calculator`:
1. (Recommended) With [`AdamiPressureExtrapolation`](@ref), the pressure is extrapolated from the pressure of the
   fluid according to [Adami et al., 2012](@cite Adami2012), and the density is obtained by applying the inverse of the state equation.
   This option usually yields the best results of the options listed here.
2. (Only relevant for FSI) With [`BernoulliPressureExtrapolation`](@ref), the pressure is extrapolated from the
   pressure similar to the [`AdamiPressureExtrapolation`](@ref), but a relative velocity-dependent pressure part
   is calculated between moving bodies and fluids, which increases the boundary pressure in areas prone to
   penetrations.
3. With [`SummationDensity`](@ref), the density is calculated by summation over the neighboring particles,
   and the pressure is computed from the density with the state equation.
4. With [`ContinuityDensity`](@ref), the density is integrated from the continuity equation,
   and the pressure is computed from the density with the state equation.
   Note that this causes a gap between fluid and boundary where the boundary is initialized
   without any contact to the fluid. This is due to overestimation of the boundary density
   as soon as the fluid comes in contact with boundary particles that initially did not have
   contact to the fluid.
   Therefore, in dam break simulations, there is a visible "step", even though the boundary is supposed to be flat.
   See also [dual.sphysics.org/faq/#Q_13](https://dual.sphysics.org/faq/#Q_13).
5. With [`PressureZeroing`](@ref), the density is set to the reference density and the pressure
   is computed from the density with the state equation.
   This option is not recommended. The other options yield significantly better results.
6. With [`PressureMirroring`](@ref), the density is set to the reference density. The pressure
   is not used. Instead, the fluid pressure is mirrored as boundary pressure in the
   momentum equation.
   This option is not recommended due to stability issues. See [`PressureMirroring`](@ref)
   for more details.

#### 1. [`AdamiPressureExtrapolation`](@ref)

The pressure of the boundary particles is obtained by extrapolating the pressure of the fluid
according to [Adami et al., 2012](@cite Adami2012).
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

#### 2. [`BernoulliPressureExtrapolation`](@ref)
Identical to the pressure ``p_b `` calculated via [`AdamiPressureExtrapolation`](@ref), but it adds the dynamic pressure component of the Bernoulli equation:
```math
p_b = \frac{\sum_f (p_f + \frac{1}{2} \, \rho_{\text{neighbor}} \left( \frac{ (\mathbf{v}_f - \mathbf{v}_{\text{body}}) \cdot (\mathbf{x}_f - \mathbf{x}_{\text{neighbor}}) }{ \left\| \mathbf{x}_f - \mathbf{x}_{\text{neighbor}} \right\| } \right)^2 \times \text{factor} +\rho_f (\bm{g} - \bm{a}_b) \cdot \bm{r}_{bf}) W(\Vert r_{bf} \Vert, h)}{\sum_f W(\Vert r_{bf} \Vert, h)}
```
where ``\mathbf{v}_f`` is the velocity of the fluid and ``\mathbf{v}_{\text{body}}`` is the velocity of the body.
This adjustment provides a higher boundary pressure for bodies moving with a relative velocity to the fluid to prevent penetration.
This modification is original and not derived from any literature source.

```@docs
    BernoulliPressureExtrapolation
```

#### 5. [`PressureZeroing`](@ref)

This is the simplest way to implement dummy boundary particles.
The density of each particle is set to the reference density and the pressure to the
reference pressure (the corresponding pressure to the reference density by the state equation).
```@docs
    PressureZeroing
```

#### 6. [`PressureMirroring`](@ref)

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

This approach was first mentioned by [Akinci et al. (2012)](@cite Akinci2012) and written down in this form
by [Band et al. (2018)](@cite Band2018a).
```@docs
    PressureMirroring
```

### No-slip conditions

For the interaction of dummy particles and fluid particles, [Adami et al. (2012)](@cite Adami2012)
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

## Repulsive Particles

Boundaries modeled as boundary particles which exert forces on the fluid particles ([Monaghan, Kajtar, 2009](@cite Monaghan2009)).
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
dimensions (see [Monaghan & Kajtar, 2009](@cite Monaghan2009), Equation (3.1) and [Valizadeh & Monaghan, 2015](@cite Valizadeh2015)).
Note that the repulsive acceleration $\tilde{f}_{ab}$ does not depend on the masses of
the boundary particles.
Here, ``\Phi`` denotes the 1D Wendland C4 kernel, normalized to ``1.77`` for ``q=0``
([Monaghan & Kajtar, 2009](@cite Monaghan2009), Section 4), with ``\Phi(r, h) = w(r/h)`` and
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
According to [Monaghan & Kajtar (2009)](@cite Monaghan2009), a value of ``\beta = 3`` for the Wendland C4 that
we use here is reasonable for most computing purposes.

The parameter ``K`` is used to scale the force exerted by the boundary particles.
In [Monaghan & Kajtar (2009)](@cite Monaghan2009), a value of ``gD`` is used for static tank simulations,
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

```@docs
    BoundaryModelMonaghanKajtar
```

# [Open Boundaries](@id open_boundary)

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "boundary", "open_boundary", "system.jl")]
```

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "boundary", "open_boundary", "boundary_zones.jl")]
```

```@autodocs
Modules = [TrixiParticles]
Filter = t -> typeof(t) === typeof(TrixiParticles.planar_geometry_to_face)
```

# [Open Boundary Models](@id open_boundary_models)
We offer two models for open boundaries, with the choice depending on the specific problem and flow characteristics near the boundary:
1. [**Method of characteristics**](@ref method_of_characteristics): The method of characteristics is typically used in problems where tracking of wave propagation
    or flow in a domain that interacts with open boundaries (e.g., shock waves, wave fronts, or any behavior that depends on the direction of propagation) is needed.
    It avoids artificial reflections that could arise from boundary conditions.
1. [**Mirroring**](@ref mirroring): The mirroring method is often applied when the flow near the boundary is expected to behave in a way that is easier to model by using symmetry
    or when the fluid does not exhibit complex wave behavior near the boundary (e.g., free-surface flows and simple outflow).

## [Method of characteristics](@id method_of_characteristics)

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "boundary", "open_boundary", "method_of_characteristics.jl")]
```

The difficulty in non-reflecting boundary conditions, also called open boundaries, is to determine
the appropriate boundary values of the exact characteristics of the Euler equations.
Assuming the flow near the boundaries is normal to the boundary
and free of shock waves and significant viscous effects, it can be shown that three characteristic variables exist:

- ``J_1``, associated with convection of entropy and propagates at flow velocity,
- ``J_2``, downstream-running characteristics,
- ``J_3``, upstream-running characteristics.

[Giles (1990)](@cite Giles1990) derived those variables based on a linearized set of governing equations:
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

[Lastiwka et al. (2009)](@cite Lastiwka2008) applied the method of characteristic to SPH and determined the number of variables that should be
**prescribed** at the boundary and the number which should be **propagated** from the fluid domain to the boundary:

- For an **inflow** boundary:
    - Prescribe *downstream*-running characteristics ``J_1`` and ``J_2``
    - Transmit ``J_3`` from the fluid domain (allow ``J_3`` to propagate upstream to the boundary).

- For an **outflow** boundary:
    - Prescribe *upstream*-running characteristic ``J_3``
    - Transmit ``J_1`` and ``J_2`` from the fluid domain.

Prescribing is done by simply setting the characteristics to zero. To transmit the characteristics from the fluid
domain, or in other words, to carry the information of the fluid to the boundaries, [Negi et al. (2020)](@cite Negi2020) use a Shepard Interpolation
```math
f_i = \frac{\sum_j^N f_j W_{ij}}{\sum_j^N W_{ij}},
```
where the ``i``-th particle is a boundary particle, ``f`` is either  ``J_1``, ``J_2`` or ``J_3`` and ``N`` is the set of
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
With ``J_1``, ``J_2`` and ``J_3`` determined, we can easily solve for the actual variables for each particle.

## [Mirroring](@id mirroring)
```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "boundary", "open_boundary", "mirroring.jl")]
```

## [Dynamical Pressure](@id dynamical_pressure)
```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "boundary", "open_boundary", "dynamical_pressure.jl")]
```

Unlike the [method of characteristics](@ref method_of_characteristics) or the [mirroring](@ref mirroring) method,
which compute the physical properties of buffer particles within the [`BoundaryZone`](@ref)
based on information from the fluid domain, this model directly solves the momentum equation for the buffer particles.

A key challenge arises from the truncated support close to the free surface within the [`BoundaryZone`](@ref).
This truncation leads to inaccurate evaluation of the pressure gradient.
To address this issue, [Zhang et al. (2025)](@cite Zhang2025) introduce an additional term
(second term in eq. (13) in [Zhang2025](@cite)) in the momentum equation:
```math
+ 2 p_b \sum_j \left( \frac{m_j}{\rho_i \rho_j} \right) \nabla W_{ij},
```
where ``p_b`` is the prescribed dynamical boundary pressure.
Note the positive sign, which compensates for the missing contribution due to the truncated support domain.
This term vanishes for particles with full kernel support.
Thus, it can be applied to all particles within the [`BoundaryZone`](@ref)
without the need to specifically identify those near the free surface.

To further handle incomplete kernel support, for example in the viscous term of the momentum equation,
the updated velocity of particles within the [`BoundaryZone`](@ref) is projected onto the face normal,
so that only the component in flow direction is kept.

# Pressure Models
```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "boundary", "open_boundary", "pressure_model.jl")]
```

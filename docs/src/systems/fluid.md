# [Fluid Models](@id fluid_models)

Currently available fluid methods are the [weakly compressible SPH method](@ref wcsph) and the
[entropically damped artificial compressibility for SPH](@ref edac).
This page lists models and techniques that apply to both of these methods.

## [Viscosity](@id viscosity_sph)

Viscosity is a critical physical property governing momentum diffusion within a fluid.
In the context of SPH, viscosity determines how rapidly velocity gradients are smoothed out,
influencing key flow characteristics such as boundary layer formation, vorticity diffusion,
and dissipation of kinetic energy. It also helps determine whether a flow is laminar or turbulent
under a given set of conditions.

Implementing viscosity correctly in SPH is essential for producing physically accurate results,
and different methods exist to capture both numerical stabilization and true viscous effects.

### Artificial (numerical) viscosity

Artificial (numerical) viscosity is a technique used to stabilize simulations of inviscid flows,
which would otherwise show unphysical particle movement due to numerical instability.
To achieve this, a dissipative term is added to the momentum equations in a way that it
does not significantly alter the physical behavior of the flow.
This approach is especially useful in simulations such as high-speed flows with strong shocks or astrophysical scenarios,
where other approaches are insufficient to stabilize the simulation.

### Physical (real) viscosity

Physical viscosity is essential for accurately modeling the true viscous stresses within a fluid.
It ensures that simulations align with a target Reynolds number or adhere to experimentally measured fluid properties.
This is achieved by incorporating forces that replicate the viscous stress term found in the Navier–Stokes equations.
As a result, the method is particularly effective for simulating low-speed, incompressible, or weakly compressible flows,
where it is crucial to capture the actual behavior of the fluid.

### Model comparison

#### ArtificialViscosityMonaghan

`ArtificialViscosityMonaghan` by Monaghan ([Monaghan1992](@cite), [Monaghan1989](@cite))
should be mainly used for inviscid flows (Euler), artificial stabilization
or shock-capturing, for which Monaghan [Monaghan1989](@cite) originally designed
this term to provide smoothing across shocks, intentionally overestimating the physical viscosity.
The implementation includes a dissipation term that becomes more significant
as particles approach one another. This helps suppress tensile instabilities,
which can lead to particle clumping and effectively smooths out high-frequency pressure fluctuations.
This increase in dissipation is triggered by the relative motion between particles:
as particles come closer and compress the local flow,
the artificial viscosity term becomes stronger to damp out rapid changes
and prevent unphysical clustering.
This ensures that while the simulation remains stable in challenging
flow regimes with large density or pressure variations,
the physical behavior is not overly altered.
Several extensions have been proposed to limit the dissipation effect for example
by Balsara ([Balsara1995](@cite)) or Morris ([Morris1997](@cite)).

##### Mathematical Formulation

The acceleration contribution from particle ``b`` to particle ``a`` is

```math
\left.\frac{\mathrm{d}\bm{v}_a}{\mathrm{d}t}\right|_{ab}^{\text{AV}} =
\begin{cases}
    m_b \frac{\alpha c \mu_{ab} + \beta \mu_{ab}^2}{\bar{\rho}_{ab}}
    \nabla_a W_{ab}, & \text{if } \bm{v}_{ab} \cdot \bm{r}_{ab} < 0, \\
    0, & \text{otherwise}.
\end{cases}
```

where:

- ``\alpha`` and ``\beta`` are viscosity parameters,
- ``c`` is the local speed of sound,
- ``\bar{\rho}_{ab}`` is the arithmetic mean of the densities of particles ``a`` and ``b``.

The term ``\mu_{ab}`` is defined as

```math
\mu_{ab} = \frac{h \, \bm{v}_{ab} \cdot \bm{r}_{ab}}
                {\Vert \bm{r}_{ab} \Vert^2 + \epsilon h^2},
```

with:

- ``h`` being the smoothing length,
- ``\epsilon`` a small parameter to prevent singularities,
- ``\bm{r}_{ab} = \bm{r}_a - \bm{r}_b`` representing the difference of the coordinate vectors,
- ``\bm{v}_{ab} = \bm{v}_a - \bm{v}_b`` representing the relative velocity between particles.

##### Resolution Dependency and Effective Viscosity

To ensure that the simulation maintains a consistent Reynolds number when the resolution changes, the parameter ``\alpha`` must be adjusted accordingly.
Monaghan (2005) introduced an effective physical kinematic viscosity ``\nu`` defined as:

```math
\nu = \frac{\alpha h c}{2d + 4},
```

where **``d``** is the number of spatial dimensions. This relation allows the calibration of ``\alpha`` to achieve the desired viscous behavior as the resolution or simulation conditions vary.

#### ViscosityMorris

`ViscosityMorris` is ideal for moderate to low Mach number flows where accurately modeling physical viscous behavior is essential.
Developed by [Morris (1997)](@cite Morris1997) and later applied by [Fourtakas (2019)](@cite Fourtakas2019),
this method directly simulates the viscous stresses found in fluids rather than relying on artificial viscosity.
By approximating momentum diffusion based on local fluid properties, the method captures the actual viscous forces without excessive damping.
This results in a more realistic representation of flow dynamics in weakly compressible scenarios.

##### Mathematical Formulation

An additional force term ``\tilde{\bm{F}}_{ab}`` is introduced in the momentum equation:

```math
\tilde{\bm{F}}_{ab} =
m_a m_b \frac{(\mu_a + \mu_b)\, \bm{r}_{ab} \cdot \nabla_a W_{ab}}
{\rho_a \rho_b (\Vert \bm{r}_{ab} \Vert^2 + \epsilon h^2)}\, \bm{v}_{ab},
```

where:

- ``\mu_a = \rho_a \nu`` and ``\mu_b = \rho_b \nu`` represent the dynamic viscosities of particles ``a`` and ``b`` (with ``\nu`` being the kinematic viscosity),
- ``\bm{r}_{ab} = \bm{r}_a - \bm{r}_b`` represents the difference of the coordinate vectors,
- ``\bm{v}_{ab} = \bm{v}_a - \bm{v}_b`` represents the relative velocity between particles,
- `` h `` is the smoothing length,
- `` \nabla_a W_{ab} `` is the gradient of the smoothing kernel,
- `` \epsilon `` is a small parameter to prevent singularities.

#### ViscosityAdami

`ViscosityAdami`, introduced by [Adami (2012)](@cite Adami2012), is optimized for incompressible or weakly compressible flows where precise modeling of shear stress is critical.
It enhances boundary layer representation by better resolving shear gradients, increasing dissipation in regions with steep velocity differences (e.g., near solid boundaries)
while minimizing compressibility effects. This results in accurate laminar flow simulations and accurate physical shear stresses.

##### Mathematical Formulation

The viscous interaction is modeled through the following pairwise force:

```math
\bm{F}_{ab}^{\nu} =
\left( V_a^2 + V_b^2 \right)\,
\bar{\eta}_{ab}\,
\frac{\nabla_a W_{ab} \cdot \bm{r}_{ab}}
{\Vert \bm{r}_{ab} \Vert^2 + \epsilon h_{ab}^2}\,
\bm{v}_{ab}.
```

where:

- `` \bm{r}_{ab} = \bm{r}_a - \bm{r}_b `` is the difference of the coordinate vectors,
- `` \bm{v}_{ab} = \bm{v}_a - \bm{v}_b `` is their relative velocity,
- `` V_a = m_a / \rho_a`` and `` V_b = m_b / \rho_b`` are the particle volumes,
- `` h_{ab} = \frac{1}{2}(h_a + h_b) `` is the arithmetic mean of the smoothing lengths,
- `` \nabla_a W_{ab} `` is the gradient of the smoothing kernel,
- `` \epsilon `` is a small parameter that prevents singularities (see [Ramachandran (2019)](@cite Ramachandran2019)).

The inter-particle-averaged shear stress is defined as:

```math
\bar{\eta}_{ab} = \frac{2 \eta_a \eta_b}{\eta_a + \eta_b},
```

with the dynamic viscosity of each particle given by `` \eta_a = \rho_a \nu_a ``, where `` \nu_a `` is the kinematic viscosity.

#### ViscosityCarreauYasuda

`ViscosityCarreauYasuda` implements the Carreau–Yasuda non-Newtonian viscosity model,
originally proposed by [Carreau (1972)](@cite Carreau1972) and extended by
[Yasuda et al. (1981)](@cite Yasuda1981). In this model, the kinematic viscosity
depends on the local shear rate. This makes it suitable for shear-thinning and
shear-thickening fluids, such as polymer solutions or blood-like fluids.
Instead of prescribing a single constant viscosity, the apparent viscosity
smoothly transitions between a low-shear plateau and a high-shear plateau.

In SPH, this can be incorporated by evaluating a shear-rate-dependent
viscosity locally and using it in the standard viscous discretization. A Newtonian
fluid is recovered as a special case when the parameters are chosen such that the
viscosity becomes independent of the shear rate. ([Zhang et al. (2017)](@cite Zhang2017);
[Vahabi & Sadeghy (2014)](@cite VahabiSadeghy2014)).


##### Mathematical Formulation

In the Carreau–Yasuda model, the kinematic viscosity ``\nu`` depends on the shear-rate magnitude ``\dot\gamma`` as
```math
\nu(\dot\gamma) = \nu_\infty + (\nu_0 - \nu_\infty)
\left[ 1 + (\lambda \dot\gamma)^a \right]^{\frac{n-1}{a}}.
```
where

- ``\nu_0``: zero-shear kinematic viscosity,
- ``\nu_\infty``: infinite-shear kinematic viscosity,
- ``\lambda``: time constant,
- ``a``: Yasuda parameter,
- ``n``: power-law index (``n < 1`` for shear-thinning, ``n > 1`` for shear-thickening),
- ``\dot\gamma``: shear-rate magnitude.

In this implementation the shear-rate magnitude is approximated per particle pair as
``\dot\gamma \approx \frac{\lVert \mathbf{v}_{ab} \rVert}{\lVert \mathbf{r}_{ab} \rVert + \epsilon}``,
with ``\mathbf{v}_{ab}`` the relative velocity, ``\mathbf{r}_{ab}`` the position difference,
and ``\epsilon`` a small regularization parameter.

All viscosities here are kinematic viscosities (m²/s); dynamic viscosity is obtained internally
via ``\eta = \rho \nu``. A Newtonian fluid is recovered for ``n = 1`` and
``\nu_0 = \nu_\infty``

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "viscosity.jl")]
```

## [Corrections](@id corrections)

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("general", "corrections.jl")]
```

---

## [Surface Normals](@id surface_normal)

### Overview of surface normal calculation in SPH

Surface normals provide the directionality of forces acting at the fluid interface. They are
used by the full Akinci model and both Morris models, but not by the cohesion-only Akinci model.
They are calculated based on the particle properties and their spatial distribution.

#### Color field and gradient-based surface normals

The surface normal at a particle is derived from the color field, a scalar field assigned to particles
to distinguish between different fluid phases or between fluid and air. The color field gradients point
towards the interface, and the normalized gradient defines the surface normal direction.

The unscaled colorfield gradient ``\bm{g}_a`` is

```math
\bm{g}_a = \sum_b m_b \frac{c_b}{\rho_b} \nabla_a W_{ab},
```

where:

- ``c_b`` is the color field value for particle ``b``,
- ``m_b`` is the mass of particle ``b``,
- ``\rho_b`` is the density of particle ``b``,
- ``\nabla_a W_{ab}`` is the gradient of the smoothing kernel ``W_{ab}`` with respect to particle ``a``.

For single-fluid surface-normal calculations, ``c_b = 1`` for neighboring fluid particles.

#### Model-specific scaling

The Morris models normalize the colorfield gradient:

```math
\hat{\bm{n}}_a = \frac{\bm{g}_a}{\Vert \bm{g}_a \Vert}.
```

The Akinci model instead uses the dimensionless normal from Equation 2 of
[Akinci et al. (2013)](@cite Akinci2013):

```math
\bm{n}^{A}_a = h_c \bm{g}_a,
```

where ``h_c`` is the compact-support radius. TrixiParticles.jl stores ``\bm{g}_a`` so the same
cache can serve both model families and applies the model-specific normalization or scaling when
the force is evaluated.

!!! note "Kernel correction and Akinci normals"
    A normalized [`KernelCorrection`](@ref) gradient must not be substituted directly into the
    Akinci colorfield sum. A correction constructed over the same neighbor set enforces partition
    of unity and therefore ``\sum_b V_b \nabla \widetilde{W}_{ab}=0`` for a constant color field,
    including at a free surface. The Akinci normal intentionally uses the nonzero
    neighborhood-deficiency signal of
    the uncorrected kernel gradient. First-order gradient correction and normal smoothing can
    alter the direction, but also alter or redistribute the normal magnitude used by the Akinci
    curvature force; they are not enabled implicitly.

#### Handling noise and errors in normal calculation

In regions distant from the interface, the calculated normals may be small or inaccurate due to the
smoothing kernel's support radius. To mitigate this:

1. Normals below a threshold are excluded from further calculations.
2. Curvature calculations use a corrected formulation to reduce errors near interface fringes.

#### Extensions beyond the published formulation

The published normal formulations, including Equation 2 of
[Akinci et al. (2013)](@cite Akinci2013), sum over fluid neighbors only. This is also the
default for `SurfaceTensionAkinci`. TrixiParticles.jl provides two optional or practical
extensions to the colorfield gradient:

1. **Wall-contact augmentation.** When the smoothed colorfield of a boundary particle exceeds
   the `boundary_contact_threshold` of [`ColorfieldSurfaceNormal`](@ref) relative to the maximum
   boundary colorfield, boundary neighbors also contribute to the gradient. The contribution
   reuses the fluid particle's own mass and density as a proxy for the missing fluid continuum.
   Fluid particles resting on a wetted wall that continues the fluid lattice are thereby treated
   like interior particles with near-zero normals, which suppresses spurious curvature forces
   along walls. Fluid that is not in contact with the wall keeps its free-surface normal.
   Set a finite `boundary_contact_threshold` explicitly to enable this extension for the Akinci
   model.
2. **Neighbor-count filter.** Normals of particles with fewer than ``2^d + 1`` neighbors in
   ``d`` dimensions are set to zero to remove noisy, underdefined gradients. For isolated
   droplets or spray, this disables only the curvature contribution of the Akinci model;
   the pairwise cohesion force remains active.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "surface_normal_sph.jl")]
```

---

## [Surface Tension](@id surface_tension)

Surface tension is a key phenomenon in fluid dynamics, influencing the behavior of droplets, bubbles, and fluid interfaces.
In SPH, surface tension is modeled as forces arising due to surface curvature and relative particle movement, ensuring realistic
simulation of capillary effects, droplet coalescence, and fragmentation.

The physical surface tension ``\sigma`` quantifies the energy required to increase the surface area
of a fluid by a unit amount. A higher value of ``\sigma`` indicates that the fluid resists changes to its surface area more strongly,
causing droplets or bubbles to assume shapes (often spherical) that minimize their surface. In practice, ``\sigma`` can be measured
experimentally through techniques such as the pendant drop method, the Wilhelmy plate method, or the du Noüy ring method,
each of which relates a measurable force or change in shape to the fluid’s surface tension. For pure substances,
tabulated reference values of ``\sigma`` at given temperatures are commonly used, while for mixtures or complex fluids,
direct experimental measurements or values can be estimated from empirical equation (see [Poling](@cite Poling2001) or [Lange](@cite Lange2005)).
In the following table some values are shown for reference. The values marked with a '~' are complex mixtures that are estimated by an empirical equation (see [Poling](@cite Poling2001)).

| **Fluid**    | **Surface Tension (``\sigma``) [N/m at 20°C]** |
|--------------|----------------------------------------------:|
| **Gasoline**    | ~0.022   [Poling](@cite Poling2001)             |
| **Ethanol**     | 0.022386 [Lange](@cite Lange2005)               |
| **Acetone**     | 0.02402  [Lange](@cite Lange2005)               |
| **Mineral Oil** | ~0.030   [Poling](@cite Poling2001)             |
| **Olive Oil**   | 0.03303  [Hui](@cite Hui1992), [MeloEspinosa](@cite MeloEspinosa2014) |
| **Glycerol**    | 0.06314  [Lange](@cite Lange2005)               |
| **Water**       | 0.07288  [Lange](@cite Lange2005)               |
| **Mercury**     | 0.486502 [Lange](@cite Lange2005)               |

### Model configuration

All surface tension coefficients must be finite and non-negative. A zero coefficient disables
the fluid-fluid surface force. `SurfaceTensionMorris`, `SurfaceTensionMomentumMorris`, and
`SurfaceTensionAkinciCohesionPhysical` accept physical values in N/m. The coefficients of
`CohesionForceAkinci` and the complete `SurfaceTensionAkinci` model retain the numerical meaning
defined in the original Akinci formulation. Wall adhesion is controlled independently by the
boundary's `adhesion_coefficient`.

`CohesionForceAkinci` and `SurfaceTensionAkinciCohesionPhysical` only evaluate pairwise cohesion
and optional wall forces. They do not require surface normals or `reference_particle_spacing`.
The complete `SurfaceTensionAkinci` model and both Morris models require a surface-normal method.
The Morris models default to `ColorfieldSurfaceNormal()`. The complete Akinci model defaults to
`ColorfieldSurfaceNormal(boundary_contact_threshold=Inf)` so that its normal follows the
published fluid-only sum; wall adhesion remains a separate pair force.

`ColorfieldSurfaceNormal(normal_smoothing=true)` applies one activity-weighted Shepard pass to the
unit-normal directions before Morris curvature or CSS stress evaluation. The interface activity and
surface delta remain those of the unsmoothed color gradient. Particle shifting also continues to use
the raw normal, so enabling capillary smoothing does not silently change the regularization model.
This is an explicit stabilization choice rather than a default.

Three-dimensional counterparts of every simulation experiment in
[Akinci et al. (2013)](@cite Akinci2013) are provided. They use reduced particle counts and,
except for the reported dimensions in the water-crown setup, reduced dimensions. They are
demonstrations rather than quantitative reproductions of the rendered reference scenes.

| Paper experiment | Runnable example |
|:-----------------|:-----------------|
| Figures 1 and 5, water crown | `examples/fluid/akinci_water_crown_3d.jl` |
| Figure 2, cube-to-sphere comparison | `examples/fluid/akinci_cube_to_sphere_3d.jl` |
| Figure 6, droplet impact on a plate | `examples/fluid/akinci_droplet_on_plate_3d.jl` |
| Figure 7, stream flowing over a sphere | `examples/fluid/akinci_stream_over_sphere_3d.jl` |
| Figure 8, wetting regimes | `examples/fluid/akinci_wetting_3d.jl` |
| Figure 9, splitting in an adhesive box | `examples/fluid/akinci_droplet_splitting_3d.jl` |
| Figure 10, rolling drop and two-way adhesion | `examples/fluid/akinci_rolling_droplet_3d.jl` |

Set `wetting_case` in the wetting example to `"no_wetting"`, `"weak_wetting"`,
`"moderate_wetting"`, `"intermediate_wetting"`, `"strong_wetting"`,
`"near_perfect_wetting"`, or `"perfect_wetting"`. These presets follow the coefficient sequence
shown in the paper's companion video.
The rolling-drop example uses rigid figure-shaped bodies in place of the paper's articulated
ragdolls while preserving the adhesive/non-adhesive comparison.

!!! note "Akinci kernels in two dimensions"
    Akinci et al. published the cohesion and adhesion kernels for three dimensions. In two
    dimensions, TrixiParticles.jl uses an integral-matching extension: each radial 2D kernel
    has the same full-space integral as its published 3D counterpart. This convention is not
    part of the original model, but gives both kernels dimensions of ``L^{-d}`` in ``d``
    dimensions. Their products with particle mass are therefore independent of resolution at
    a fixed smoothing-length-to-spacing ratio. Akinci surface tension is supported in two and
    three dimensions only.

    To preserve the pairwise cohesion and adhesion contributions from a previous 2D
    configuration that used the 3D normalizations, scale the coefficients at its
    compact-support radius ``h_c`` as

    ```math
    \sigma_{\mathrm{new}} = \frac{627}{790h_c}\sigma_{\mathrm{old}}, \qquad
    \beta_{\mathrm{new}} = \frac{42}{65h_c}\beta_{\mathrm{old}}.
    ```

    The migrated coefficients can then be held fixed when changing the resolution. Since
    `SurfaceTensionAkinci` uses ``\sigma`` for both cohesion and the unchanged curvature term,
    this migration also changes their relative weight; full-model configurations may require
    additional calibration.

### [Akinci-based intra-particle force surface tension and wall adhesion model](@id akinci_ipf)

The [Akinci](@cite Akinci2013) model divides surface tension into distinct force components,
which TrixiParticles.jl applies as acceleration contributions.

#### Cohesion contribution

The cohesion contribution captures the attraction between particles at the fluid interface, creating the effect of surface tension.
It is defined by the distance between particles and the support radius ``h_c``, using a kernel-based formulation.

**Key features:**

- Particles within half the support radius experience a repulsive force to prevent clustering.
- Particles beyond half the radius but within the support radius experience an attractive force to simulate cohesion.

In the acceleration form used by TrixiParticles.jl, the pairwise cohesion contribution is

```math
\left.\frac{\mathrm{d}\bm{v}_a}{\mathrm{d} t}\right|_{ab}^{\text{cohesion}}
= -\sigma m_b C_d(r) \frac{\bm{r}}{\Vert \bm{r} \Vert},
```

where the dimension-dependent cohesion kernel is

```math
C_d(r)=\frac{K_d}{h_c^{d+6}}
\begin{cases}
(h_c-r)^3 r^3, & \text{if } 2r > h_c, \\
2(h_c-r)^3 r^3 - \frac{h_c^6}{64}, & \text{if } r > 0 \text{ and } 2r \leq h_c, \\
0, & \text{otherwise.}
\end{cases}
\qquad
K_2=\frac{25280}{627\pi}, \quad K_3=\frac{32}{\pi}.
```

The 3D constant is the published normalization. The 2D constant is chosen such that

```math
\int_{\mathbb{R}^2} C_2(\Vert\bm{r}\Vert)\,\mathrm{d}A
= \int_{\mathbb{R}^3} C_3(\Vert\bm{r}\Vert)\,\mathrm{d}V
= \frac{79}{336}.
```

#### Physical cohesion-only normalization

[`SurfaceTensionAkinciCohesionPhysical`](@ref) converts a physical surface tension ``\sigma``
in N/m to the coefficient of the three-dimensional cohesion kernel. For a planar interface,

```math
\int_0^{h_c} r^4 C_3(r)\,\mathrm{d}r = \frac{21h_c^2}{880\pi},
\qquad
\sigma = \frac{\pi}{8}\gamma\rho_0^2
         \int_0^{h_c} r^4 C_3(r)\,\mathrm{d}r
       = \frac{21}{7040}\gamma\rho_0^2h_c^2.
```

The internal coefficient is therefore

```math
\gamma(h_c) = \frac{7040\sigma}{21\rho_0^2h_c^2}.
```

This support-radius scaling removes the resolution dependence of the continuum surface energy.
The model uses only central pair forces, needs no colorfield normals, and exactly conserves
pairwise linear and angular momentum. It is restricted to three dimensions because the physical
surface-energy conversion above is three-dimensional. Use [`CohesionForceAkinci`](@ref) for an
empirical coefficient or a two-dimensional simulation.

For this model, a wall's `adhesion_coefficient` multiplies the same normalized cohesion kernel.
The Young-Dupre work-of-adhesion relation gives

```math
\texttt{adhesion_coefficient} = \frac{1 + \cos\theta}{2}.
```

Thus, ratios `0`, `0.5`, and `1` target contact angles of 180, 90, and 0 degrees, respectively.
The ratio belongs to the boundary, so different walls can use different wetting properties.

Automatic time-step selection applies the capillary condition

```math
\Delta t_\sigma = \sqrt{\frac{\rho_0 h^3}{2\pi\sigma}},
```

where ``h`` is the smoothing length. If [`AkinciFreeSurfaceCorrection`](@ref) is also configured,
its local density factor multiplies the pair force. In that case, ``\sigma`` still normalizes the
underlying continuum cohesion potential, while the corrected coarse-resolution force need not
equal an independently fitted Laplace-pressure coefficient.

#### Surface area minimization contribution

The surface area minimization contribution models curvature reduction and acts on the
difference in the dimensionless Akinci normals:
```math
\left.\frac{\mathrm{d}\bm{v}_a}{\mathrm{d} t}\right|_{ab}^{\text{curvature}}
= -\sigma (\bm{n}^{A}_a - \bm{n}^{A}_b)
= -\sigma h_c (\bm{g}_a - \bm{g}_b),
```
for equal compact-support radii. Here ``h_c`` is the compact-support radius used in the Akinci
paper, not the smaller smoothing-length parameter accepted by kernels whose support spans a
multiple of that parameter.

!!! warning "Constant smoothing length"
    The pairwise force evaluates the support radius and both normals with the smoothing length
    of the first particle of the pair. With a per-particle (variable) smoothing length, the
    pair force would lose its antisymmetry and no longer conserve momentum. Use a constant
    smoothing length with `SurfaceTensionAkinci`.

#### Combined-force correction

To compensate for particle-neighborhood deficiency at a free surface, the cohesion and
curvature contributions are multiplied by the symmetric factor

```math
K_{ab} = \frac{2\rho_0}{\rho_a + \rho_b}.
```

[`AkinciFreeSurfaceCorrection`](@ref) implements this factor for the combined fluid-fluid
surface tension force. Section 4 of [Akinci et al. (2013)](@cite Akinci2013) also applies the
factor to viscosity for the same particle-deficiency reason. It does not modify pressure or wall
adhesion forces.

The published correction assumes that the density estimate reflects missing neighbors. With
[`SummationDensity`](@ref), ``\rho_a`` and ``\rho_b`` in ``K_{ab}`` are the current densities.
For [`ContinuityDensity`](@ref) in a [`WeaklyCompressibleSPHSystem`](@ref), TrixiParticles.jl
reconstructs the auxiliary densities

```math
\widetilde{\rho}_a = \sum_b m_b W_{ab}
```

and uses ``\widetilde{\rho}_a`` and ``\widetilde{\rho}_b`` only in ``K_{ab}``. Pressure and all
other density-dependent terms continue to use the integrated continuity density. The auxiliary
sum includes dummy boundary particles, so a wall that completes the particle neighborhood is not
misclassified as a free surface. This extension makes the correction independent of the selected
density calculator at the cost of one additional density-summation neighbor loop per update stage.

#### Wall adhesion contribution

This contribution models the interaction between fluid and solid boundaries, simulating adhesion effects at walls.
It uses a custom kernel with a peak at 0.75 times the support radius:

```math
\left.\frac{\mathrm{d}\bm{v}_a}{\mathrm{d} t}\right|_{ab}^{\text{adhesion}}
= -\beta m_b A_d(r) \frac{\bm{r}}{\Vert \bm{r} \Vert},
```

where the dimension-dependent adhesion kernel is

```math
A_d(r) = \frac{b_d}{h_c^{d+1/4}}
\begin{cases}
\sqrt[4]{-\frac{4r^2}{h_c} + 6r - 2h_c}, & \text{if } 2r > h_c \text{ and } r \leq h_c, \\
0, & \text{otherwise.}
\end{cases}
\qquad
b_2=\frac{13}{1200}, \quad b_3=0.007.
```

Again, ``b_3`` is the published value and ``b_2`` matches the full-space integrals. In terms
of the dimensionless radial moments

```math
J_d = \int_{1/2}^{1} q^{d-1}\left[2(1-q)(2q-1)\right]^{1/4}\,\mathrm{d}q,
```

the beta-function identities

```math
J_2 = \frac{3}{8}B\!\left(\frac{5}{4},\frac{5}{4}\right), \qquad
J_3 = \frac{65}{224}B\!\left(\frac{5}{4},\frac{5}{4}\right)
```

give ``J_3/J_2=65/84`` and thus ``b_2=2b_3J_3/J_2=13/1200``.

---

### [Morris surface tension model](@id morris_csf)

The method described by [Morris](@cite Morris2000) estimates curvature by combining particle color gradients (see [`surface_normal`](@ref)) and smoothing functions to derive surface normals.
The computed curvature is then used to determine forces acting perpendicular to the interface.
While this method provides accurate surface tension forces, it does not explicitly conserve momentum.

In the Morris model, surface tension is computed from local interface curvature ``\kappa``, the
unit surface normal ``\hat{\bm{n}}``, and the surface delta ``\delta_s``. The acceleration is a
particle-local source evaluated once per RHS:

```math
\frac{\mathrm d\bm v_a}{\mathrm dt}\bigg|_\sigma
= -\frac{\sigma}{\rho_a}\kappa_a\delta_{s,a}\hat{\bm n}_a.
```

The factors have dimensions ``[\sigma]=kg/s^2``, ``[\kappa]=1/m``,
``[\delta_s]=1/m``, and ``[\rho]=kg/m^3``, giving acceleration in ``m/s^2``. Curvature uses
activity-weighted neighboring unit normals, so particles enter and leave the interface stencil
continuously. This formulation focuses directly on interface geometry but does not explicitly
conserve momentum; accurately estimating curvature still requires adequate resolution.

[`CorrectedCSFSurfaceNormal`](@ref) selects the corrected continuous-surface-force (C-CSF)
interface geometry of [Vergnaud et al.](@cite Vergnaud2022) for [`SurfaceTensionMorris`](@ref). It
computes the outward normal from the renormalized gradient of the minimum eigenvalue of the
first-order kernel moment. Curvature uses a renormalized divergence and the published thin-jet
angular filter; the surface delta uses the published Shepard correction.

```julia
surface_tension = SurfaceTensionMorris(surface_tension_coefficient=0.072)
surface_normal_method = CorrectedCSFSurfaceNormal()
```

This explicit opt-in supports one fluid system. Setting a finite contact angle enables the planar
boundary-integral moment, eigenvalue-gradient, color-gradient, and curvature terms together with the
distance-weighted contact-normal correction of Vergnaud et al.:

```julia
surface_normal_method = CorrectedCSFSurfaceNormal(contact_angle=90.0)
```

Every participating wall must use dummy particles with explicit positive surface quadrature weights
on its physical face (`surface_measure` in `BoundaryModelDummyParticles`) and normal offset vectors
in its `InitialCondition`. The analytical half-space overlap and boundary terms currently require a
3D `WendlandC2Kernel`. General curved BIM faces and ghost-particle C-CSF geometry are not implemented.
These boundary integrals correct C-CSF interface geometry only; continuity, pressure, and viscosity
at the wall still use the selected dummy-particle boundary model rather than the hydrodynamic BIM
equations of Vergnaud et al.

---

### [Morris-based momentum-conserving surface tension model](@id moriss_css)

[`SurfaceTensionMomentumMorris`](@ref) implements a balanced continuum-surface-stress (CSS)
formulation for one-phase free surfaces. It treats surface tension as the divergence of a
localized tangential stress, avoiding the explicit curvature pass required by
[`SurfaceTensionMorris`](@ref).

#### Stress tensor formulation

The surface tension force can be written as the divergence of a stress tensor ``\bm{S}``:

```math
\bm{F}_{a}^{\sigma} = m_a \nabla \cdot \bm{S},
```

with

```math
\bm{S} = \sigma \delta_s (I - \hat{\bm{n}} \otimes \hat{\bm{n}}).
```

with:

- ``\delta_s``: Surface delta function,
- ``\hat{\bm{n}}``: Unit normal vector,
- ``I``: Identity matrix.

For a free surface represented only by fluid particles, the raw color gradient ``\bm{g}_a`` is
sampled over one half of the kernel-smoothed interface. TrixiParticles.jl therefore stores

```math
\delta_{s,a} = 2\Vert\bm{g}_a\Vert\lambda_a,
\qquad
\hat{\bm{n}}_a = \frac{\bm{g}_a}{\Vert\bm{g}_a\Vert}.
```

The factor two makes the represented half-interface delta integrate to one. During the same
neighbor pass, the scalar consistency measure

```math
q_a = -\frac{1}{d}\sum_b \frac{m_b}{\rho_b}
      \bm{r}_{ab}\mathbin{\cdot}\nabla_a W_{ab}
```

is accumulated. The symmetric pair correction ``c_{ab}=2/(q_a+q_b)`` restores the linear
kernel-gradient scaling near truncated support while retaining an antisymmetric pair force. The
acceleration is evaluated directly, without constructing ``\bm{S}`` in memory:

```math
\frac{\mathrm{d}\bm{v}_a}{\mathrm{d}t}\bigg|_\sigma
= \sigma\sum_b\frac{m_b c_{ab}}{\rho_a\rho_b}
\left[
\delta_{s,a}(I-\hat{\bm{n}}_a\otimes\hat{\bm{n}}_a)
+\delta_{s,b}(I-\hat{\bm{n}}_b\otimes\hat{\bm{n}}_b)
\right]\nabla_a W_{ab}.
```

For constant smoothing length, the coefficient multiplying each pair is symmetric, so the model
conserves linear momentum to roundoff. The Akinci free-surface correction is not applied to this
continuum stress.

#### Smooth interface activity

Morris CSF and CSS use the same C1 interface activity ``\lambda_a``. Let
``\gamma_a=h_c\Vert\bm g_a\Vert``, ``\epsilon_n`` be `interface_threshold`, and
``\alpha`` be `interface_taper_start` (default `0.8`). With

```math
S(x)=\begin{cases}
0,&x\le0,\\
3x^2-2x^3,&0<x<1,\\
1,&x\ge1,
\end{cases}
\qquad
\lambda_{g,a}=S\!\left(
\frac{\gamma_a-\alpha\epsilon_n}{(1-\alpha)\epsilon_n}\right),
```

the gradient contribution is zero below ``\alpha\epsilon_n`` and exactly recovers the untapered
model at and above ``\epsilon_n``. The support moment ``q_a`` has continuum interior value one,
following

```math
0=\int\nabla\mathbin{\cdot}(\bm rW)\,\mathrm dV
=d\int W\,\mathrm dV+\int\bm r\mathbin{\cdot}\nabla W\,\mathrm dV.
```

For `ideal_density_threshold = tau > 0`, activity remains one through ``q_a=\tau`` and tapers
to zero over `support_taper_width` (default `0.025`):

```math
\lambda_{q,a}=1-S\!\left(\frac{q_a-\tau}{\Delta q}\right),
\qquad
\lambda_a=\lambda_{g,a}\lambda_{q,a}.
```

Setting `ideal_density_threshold=0` disables support filtering. This keyword previously compared
an integer neighbor count with an ideal count; it now represents a continuous fraction of complete
kernel support. Existing validation configurations using `0.9` migrate to `0.95`. Dummy boundary
particles complete ``q_a`` near walls, suppressing false wall-bulk interfaces without carrying
capillary stress themselves.

#### Wetted-area contact angle

Young's wall energy can be enabled explicitly through the normal method:

```julia
normal_method = ColorfieldSurfaceNormal(
    contact_model=WettedAreaContactAngle(60.0))
surface_tension = SurfaceTensionMomentumMorris(surface_tension_coefficient=0.072)
```

The model discretizes ``E_w=-\sigma\cos(\theta)A_w`` from the fluid colorfield sampled on the
physical boundary surface. Its complete derivative contains an explicit fluid-wall kernel term and
a symmetric density-conjugate term consistent with `ContinuityDensity`. Fixed walls cache the
equal-and-opposite reaction; rigid-body reactions also enter `force_per_particle`, preserving force
and torque transfer.

Each dummy boundary model must receive a nonnegative `surface_measure` vector. Positive entries are
the physical surface quadrature areas and zero entries mark deeper dummy-particle layers. The same
particles require `InitialCondition.normals`; the magnitude of each normal is the offset from the
dummy particle to the represented physical wall. Surface measures make the quadrature independent
of global orientation and can represent curved, prescribed-motion, and rigid surfaces.

The validated initial configuration is intentionally strict: 3D WCSPH or EDAC,
`ContinuityDensity`, `SurfaceTensionMomentumMorris`, `WendlandC2Kernel{3}`, `h/dx=1.4`, one fluid,
and one or more dummy-particle wall or rigid-body systems. Each boundary must contain one connected,
disk-like wetted patch. Targets must lie strictly inside `(0, 180)` degrees; `90` degrees produces
exactly zero wall energy, force, and reaction. Unsupported dimensions, kernels, ratios, density
formulations, colors, or disconnected patches raise an `ArgumentError`. Constructing
`ColorfieldSurfaceNormal()` without a contact model remains the no-wetting default.

#### Performance and limitations

The model needs one color-gradient neighbor pass. The stress divergence and wetted-area density
force are fused into the existing fluid interaction, while the explicit derivative is fused into
the fluid-boundary interaction. Per particle, the base CSS model caches the normal, surface delta,
interface activity, and one scalar correction; the full tensor is reconstructed only for VTK
output. Wetted-area fluid caches hold the density conjugate and aggregate energy/area diagnostics.
Boundary caches hold the immutable quadrature, transient area weights, and thread-local reaction
reduction storage. This is cheaper and less noisy than explicit-curvature CSF but more expensive
than empirical cohesion forces. Particle regularization or sufficient physical viscosity is
recommended for long simulations because tangential surface stress can expose the usual SPH
tensile instability. Supported Morris/CSS free-surface simulations can opt into
[`InterfaceAwareTensileInstabilityControl`](@ref), which applies TIC in the fluid interior and
blends back to conservative pressure at the interface.

The formulation currently targets one-phase free surfaces, not resolved two-phase interfaces. A
constant smoothing length is required for exact pairwise momentum conservation. Coarse droplets
with only a few support radii across the diameter can overpredict the dynamic Laplace pressure;
the error decreases under particle refinement.

### API

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "surface_tension.jl")]
```

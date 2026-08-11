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

In the literature, the unnormalized surface normal ``\bm{n}_a`` is commonly written as

```math
\bm{n}_a = \sum_b m_b \frac{c_b}{\rho_b} \nabla_a W_{ab},
```

where:

- ``c_b`` is the color field value for particle ``b``,
- ``m_b`` is the mass of particle ``b``,
- ``\rho_b`` is the density of particle ``b``,
- ``\nabla_a W_{ab}`` is the gradient of the smoothing kernel ``W_{ab}`` with respect to particle ``a``.

For single-fluid surface-normal calculations, ``c_b = 1`` for neighboring fluid particles.

#### Normalization of surface normals

The calculated normals are normalized to unit vectors:

```math
\hat{\bm{n}}_a = \frac{\bm{n}_a}{\Vert \bm{n}_a \Vert}.
```

Normalization ensures that the magnitude of the normals does not bias the curvature calculations or the resulting surface tension forces.

#### Handling noise and errors in normal calculation

In regions distant from the interface, the calculated normals may be small or inaccurate due to the
smoothing kernel's support radius. To mitigate this:

1. Normals below a threshold are excluded from further calculations.
2. Curvature calculations use a corrected formulation to reduce errors near interface fringes.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "surface_normal_sph.jl")]
```

---

## [Surface Tension](@id surface_tension)

Surface tension is a key phenomenon in fluid dynamics, influencing the behavior of droplets, bubbles, and fluid interfaces.
In SPH, surface tension is modeled as forces arising due to surface curvature and relative particle movement, ensuring realistic
simulation of capillary effects, droplet coalescence, and fragmentation.

The surface tension coefficient ``\sigma`` is a physical parameter that quantifies the energy required to increase the surface area
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
the fluid-fluid surface force. Wall adhesion is controlled independently by the boundary's
`adhesion_coefficient`.

`CohesionForceAkinci` only evaluates the pairwise cohesion and optional wall-adhesion forces.
It does not require surface normals or `reference_particle_spacing`. The full
`SurfaceTensionAkinci` model and both Morris models require a surface-normal method. When one
of these models is selected without an explicit method, `ColorfieldSurfaceNormal()` is used.

`SurfaceTensionAkinci(reference_smoothing_length=h_ref)` enables an empirical normalization of
the normal-difference force for resolution studies. The reference length must remain fixed when
changing particle spacing. Omitting it preserves the original Akinci discretization.

!!! note "Akinci kernels in two dimensions"
    Akinci et al. published the cohesion and adhesion kernels for three dimensions. In two
    dimensions, TrixiParticles.jl uses an integral-matching extension: each radial 2D kernel
    has the same full-space integral as its published 3D counterpart. This convention is not
    part of the original model, but gives both kernels dimensions of ``L^{-d}`` in ``d``
    dimensions. Their products with particle mass are therefore independent of resolution at
    a fixed smoothing-length-to-spacing ratio. Akinci surface tension is supported in two and
    three dimensions only. Integral matching removes the resolution dependence of the pair
    kernels, but does not turn their numerical coefficients into physical values in N/m.

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

#### Surface area minimization contribution

The surface area minimization contribution models curvature reduction and acts on the
difference in surface normals. TrixiParticles.jl uses the local smoothing length:

```math
\left.\frac{\mathrm{d}\bm{v}_a}{\mathrm{d} t}\right|_{ab}^{\text{curvature}}
= -\sigma h_a (\bm{n}_a - \bm{n}_b),
```

where ``\bm{n}_a`` and ``\bm{n}_b`` are the surface normals of the interacting particles.

When `reference_smoothing_length=h_ref` is specified, the normal contribution instead becomes

```math
\left.\frac{\mathrm{d}\bm{v}_a}{\mathrm{d} t}\right|_{ab}^{\text{curvature}}
= -\sigma \frac{m_b}{\bar{\rho}_{ab} h_{ab}^d} h_{\mathrm{ref}}
  (\bm{n}_a - \bm{n}_b).
```

Here ``\bar{\rho}_{ab}=(\rho_a+\rho_b)/2`` and ``h_{ab}=\min(h_a,h_b)``. This
symmetric pair normalization preserves linear momentum for unequal particle masses. The
dimensionless volume factor removes the direct dependence on neighbor count, while the fixed
``h_{\mathrm{ref}}`` prevents this contribution from weakening when ``h`` is refined. This is a
baseline-preserving empirical normalization, not a conversion of the Akinci coefficient to a
physical value in N/m.

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
For [`ContinuityDensity`](@ref) in a [`WeaklyCompressibleSPHSystem`](@ref) or
[`EntropicallyDampedSPHSystem`](@ref), TrixiParticles.jl
reconstructs the auxiliary densities

```math
\widetilde{\rho}_a = \sum_b m_b W_{ab}
```

and uses ``\widetilde{\rho}_a`` and ``\widetilde{\rho}_b`` in ``K_{ab}``. For
[`SurfaceTensionAkinci`](@ref), the same reconstructed densities are used in the particle-volume
weights ``m_b / \rho_b`` of the color-field normals, matching the density-summation formulation
of the published model. Pressure and all other density-dependent terms continue to use the
integrated continuity density. The auxiliary sum includes dummy boundary particles, so a wall
that completes the particle neighborhood is not misclassified as a free surface. This extension
makes the correction independent of the selected density calculator at the cost of one additional
density-summation neighbor loop per update stage.

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
particle-local source evaluated once per right-hand side evaluation:

```math
\frac{\mathrm d\bm v_a}{\mathrm dt}\bigg|_\sigma
= -\frac{\sigma}{\rho_a}\kappa_a\delta_{s,a}\hat{\bm n}_a.
```

The factors have dimensions ``[\sigma]=kg/s^2``, ``[\kappa]=1/m``,
``[\delta_s]=1/m``, and ``[\rho]=kg/m^3``, giving acceleration in ``m/s^2``. This formulation
does not explicitly conserve momentum, and accurately estimating curvature still requires
adequate resolution.

#### Smooth colorfield interface activity

With `ColorfieldSurfaceNormal`, Morris CSF and CSS use a C1 interface activity ``\lambda_a``. Let
``h_c`` be the compact-support radius, ``\gamma_a=h_c\Vert\bm g_a\Vert``, ``\epsilon_n`` be
`interface_threshold`, and ``\alpha`` be `interface_taper_start` (default `0.8`). With

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

the gradient contribution is zero below ``\alpha\epsilon_n`` and recovers the untapered model at
and above ``\epsilon_n``. During the normal pass, the continuous support moment

```math
q_a=-\frac{1}{d}\sum_b\frac{m_b}{\rho_b}
    \bm r_{ab}\mathbin{\cdot}\nabla_aW_{ab}
```

is accumulated without another neighbor traversal. For `ideal_density_threshold = tau > 0`, the
support activity is ``\lambda_{q,a}=1-S((q_a-\tau)/\Delta q)``, where ``\Delta q`` is
`support_taper_width` (default `0.025`). The final activity and one-phase surface delta are

```math
\lambda_a=\lambda_{g,a}\lambda_{q,a},\qquad
\delta_{s,a}=2\Vert\bm g_a\Vert\lambda_a.
```

Setting `ideal_density_threshold=0` disables support filtering. For Morris CSF/CSS with
`ColorfieldSurfaceNormal`, this keyword represents a continuous fraction of complete kernel
support instead of an integer neighbor-count fraction. Dummy boundary particles complete
``q_a`` near walls without carrying capillary stress. These controls do not alter the separate
C-CSF geometry described below.

#### Optional normal smoothing

`ColorfieldSurfaceNormal(normal_smoothing=true)` applies one activity-weighted Shepard pass to
the unit-normal directions. For active neighbors,

```math
\widetilde{\bm n}_a =
\frac{\sum_b \lambda_b(m_b/\rho_b)W_{ab}\hat{\bm n}_b}
     {\sum_b \lambda_b(m_b/\rho_b)W_{ab}},
\qquad
\hat{\bm n}^{\sigma}_a =
\frac{\widetilde{\bm n}_a}{\Vert\widetilde{\bm n}_a\Vert}.
```

The raw normal is used as a finite fallback when the weighted direction is undefined. Smoothing
changes only the normal used by Morris curvature and force evaluation and by CSS stress
evaluation. The raw colorfield normal, interface activity, and surface delta remain unchanged,
so geometry diagnostics and particle regularization are not silently modified. The option is
disabled by default and therefore adds no cache or traversal unless requested.

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
boundary-integral moment, eigenvalue-gradient, color-gradient, and curvature terms together with
the distance-weighted contact-normal correction of Vergnaud et al.:

```julia
surface_normal_method = CorrectedCSFSurfaceNormal(contact_angle=90.0)
```

The angle is measured between the wall normal pointing into the fluid and the outward interface
normal. Every participating boundary must be a stationary `WallBoundarySystem` using dummy
particles with explicit positive surface quadrature weights on its physical face
(`surface_measure` in `BoundaryModelDummyParticles`). Its `InitialCondition` normal vectors point
from each physical face into the dummy-particle layer, with magnitude equal to the face-to-particle
offset. The analytical half-space overlap and boundary terms currently require a 3D
`WendlandC2Kernel`. Moving walls, general curved BIM faces, rigid boundaries, and ghost-particle
C-CSF geometry are not implemented. These boundary integrals correct C-CSF interface geometry
only; continuity, pressure, and viscosity at the wall still use the selected dummy-particle
boundary model rather than the hydrodynamic BIM equations of Vergnaud et al.

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
sampled over one half of the kernel-smoothed interface. TrixiParticles.jl therefore uses

```math
\delta_{s,a} = 2\Vert\bm{g}_a\Vert\lambda_a,
\qquad
\hat{\bm{n}}_a = \frac{\bm{g}_a}{\Vert\bm{g}_a\Vert},
```

where ``\lambda_a`` is the smooth interface activity described above. During the same neighbor
pass, the scalar consistency measure

```math
q_a = -\frac{1}{d}\sum_b \frac{m_b}{\rho_b}
      \bm{r}_{ab}\mathbin{\cdot}\nabla_a W_{ab}
```

is accumulated. Its continuum interior value is one. The symmetric pair correction
``c_{ab}=2/(q_a+q_b)`` restores the linear kernel-gradient scaling near truncated support while
retaining an antisymmetric pair force. The acceleration is evaluated directly, without storing
``\bm S``:

```math
\frac{\mathrm{d}\bm{v}_a}{\mathrm{d}t}\bigg|_\sigma
= \sigma\sum_b\frac{m_b c_{ab}}{\rho_a\rho_b}
\left[
\delta_{s,a}(I-\hat{\bm{n}}_a\otimes\hat{\bm{n}}_a)
+\delta_{s,b}(I-\hat{\bm{n}}_b\otimes\hat{\bm{n}}_b)
\right]\nabla_a W_{ab}.
```

For constant smoothing length, every coefficient multiplying a particle pair is symmetric and
``\nabla_bW_{ba}=-\nabla_aW_{ab}``. The model therefore conserves linear momentum to roundoff.
Dummy boundary particles complete ``q_a`` near walls but do not carry capillary stress. The
Akinci free-surface correction is deliberately not applied to this continuum stress.

#### Validation

The drivers in `validation/surface_tension_2d` and `validation/surface_tension_3d` measure the
CSS operator against the Young-Laplace pressure jump. The two-dimensional driver also compares
the instantaneous mode-2 restoring stiffness with the Rayleigh frequency. Committed reference
results cover four resolutions in each dimension, while reduced CI checks require the fitted
surface-tension error to remain below 6% in 2D and 2% in 3D, the Rayleigh frequency error below
5%, and the net capillary force at roundoff.

#### Wetted-area contact angle

Young's wall energy can be enabled explicitly through the normal method:

```julia
surface_normal_method = ColorfieldSurfaceNormal(
    contact_model=WettedAreaContactAngle(60.0))
surface_tension = SurfaceTensionMomentumMorris(surface_tension_coefficient=0.072)
```

The model discretizes ``E_w=-\sigma\cos(\theta)A_w`` from the fluid colorfield sampled on the
physical boundary surface. Its complete derivative contains an explicit fluid-wall kernel term
and a symmetric density-conjugate term consistent with `ContinuityDensity`. Fixed walls cache the
equal-and-opposite reaction; rigid-body reactions also enter `force_per_particle`, preserving
force and torque transfer.

Each dummy boundary model must receive a nonnegative `surface_measure` vector. Positive entries
are physical surface quadrature areas and zero entries mark deeper dummy-particle layers. The same
particles require `InitialCondition.normals`; each normal's magnitude is the offset from the dummy
particle to the represented physical wall. This makes the quadrature independent of global
orientation and supports prescribed-motion and rigid surfaces.

The supported initial configuration is intentionally strict: 3D WCSPH or EDAC,
`ContinuityDensity`, `SurfaceTensionMomentumMorris`, `WendlandC2Kernel{3}`, `h/dx=1.4`, one fluid,
and one or more dummy-particle wall or rigid-body systems. Each boundary must contain one
connected, disk-like wetted patch. Targets must lie strictly inside `(0, 180)` degrees; `90`
degrees produces exactly zero wall energy, force, and reaction. Unsupported dimensions, kernels,
ratios, density formulations, colors, or disconnected patches raise an `ArgumentError`.
Constructing `ColorfieldSurfaceNormal()` without a contact model remains the no-wetting default.

The density force is fused into the existing fluid interaction and the explicit derivative into
the fluid-boundary interaction. Per-fluid caches hold one density-conjugate scalar per particle
and aggregate energy/area diagnostics. Boundary caches hold the immutable quadrature, transient
area weights, and reaction-reduction storage. The formulation currently targets one-phase free
surfaces and requires constant smoothing length for exact pairwise momentum conservation.

VTK output exposes both `surf_normal`, the raw geometry normal, and `surface_tension_normal`, the
normal actually used by the capillary operator. It also includes `surface_delta`,
`interface_activity`, and `surface_tension`. Morris CSF adds `curvature` and
`surface_support_moment`; CSS adds `surface_divergence_correction` and the physical
`surface_stress_tensor`, reconstructed only while writing output.

Supported Morris CSF/CSS free-surface simulations can opt into
[`InterfaceAwareTensileInstabilityControl`](@ref), which applies tensile-instability control in
the fluid interior and blends back to conservative pressure across the interface transition.

### API

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "surface_tension.jl")]
```

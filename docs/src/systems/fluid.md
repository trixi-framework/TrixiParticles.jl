# [Fluid Models](@id fluid_models)

Currently available fluid methods are the [weakly compressible SPH method](@ref wcsph) and the
[entropically damped artificial compressibility for SPH](@ref edac).
This page lists models and techniques that apply to both of these methods.

!!! note
    The formulas on this page follow the force notation commonly used in the SPH literature.
    TrixiParticles.jl usually evaluates the corresponding acceleration contributions internally.
    Whenever the implemented discretization differs from the literature formula, both forms are
    stated explicitly.

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

The force exerted by particle ``b`` on particle ``a`` due to artificial viscosity is given by

```math
\bm{F}_{ab}^{\text{AV}} = - m_a m_b \Pi_{ab} \nabla_a W_{ab}.
```

where:

- ``\Pi_{ab}`` is the artificial viscosity term defined as:
  ```math
  \Pi_{ab} =
  \begin{cases}
      -\frac{\alpha c \mu_{ab} + \beta \mu_{ab}^2}{\bar{\rho}_{ab}} & \text{if } \mathbf{v}_{ab} \cdot \mathbf{r}_{ab} < 0, \\
      0 & \text{otherwise}
  \end{cases}
  ```
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

Surface normals are essential for modeling surface tension as they provide the directionality
of forces acting at the fluid interface. They are calculated based on the particle properties and
their spatial distribution.

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

Implementation note: for the single-fluid surface-normal calculation in TrixiParticles.jl,
this reduces to
```math
\bm{n}_a = \sum_b m_b \frac{1}{\rho_b} \nabla_a W_{ab},
```
i.e. effectively ``c_b = 1`` for neighboring fluid particles.

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

### [Akinci-based intra-particle force surface tension and wall adhesion model](@id akinci_ipf)

The [Akinci](@cite Akinci2013) model divides surface tension into distinct force components:

#### Cohesion force

The cohesion force captures the attraction between particles at the fluid interface, creating the effect of surface tension.
It is defined by the distance between particles and the support radius ``h_c``, using a kernel-based formulation.

**Key features:**

- Particles within half the support radius experience a repulsive force to prevent clustering.
- Particles beyond half the radius but within the support radius experience an attractive force to simulate cohesion.

The pairwise cohesion force is

```math
\bm{F}_{\text{cohesion}} = -\sigma m_b C(r) \frac{\bm{r}}{\Vert \bm{r} \Vert},
```

where ``C(r)``, the cohesion kernel, is defined as:

```math
C(r)=\frac{32}{\pi h_c^9}
\begin{cases}
(h_c-r)^3 r^3, & \text{if } 2r > h_c, \\
2(h_c-r)^3 r^3 - \frac{h_c^6}{64}, & \text{if } r > 0 \text{ and } 2r \leq h_c, \\
0, & \text{otherwise.}
\end{cases}
```

#### Surface area minimization force

The surface area minimization force models curvature reduction and acts on the
difference in surface normals:

```math
\bm{F}_{\text{curvature}} = -\sigma (\bm{n}_a - \bm{n}_b),
```

where ``\bm{n}_a`` and ``\bm{n}_b`` are the surface normals of the interacting particles.
Implementation note: TrixiParticles.jl uses
```math
\left.\frac{\mathrm{d}\bm{v}_a}{\mathrm{d} t}\right|_{ab}^{\text{curvature}}
= -\sigma h (\bm{n}_a - \bm{n}_b),
```
which uses the same normal-difference direction but scales the magnitude with the
local smoothing length ``h``. For constant ``h``, this factor can be absorbed into
the coefficient ``\sigma``; for variable smoothing lengths, it makes the curvature
contribution depend on the local kernel support.

#### Wall adhesion force

This force models the interaction between fluid and solid boundaries, simulating adhesion effects at walls.
It uses a custom kernel with a peak at 0.75 times the support radius:

```math
\bm{F}_{\text{adhesion}} = -\beta m_b A(r) \frac{\bm{r}}{\Vert \bm{r} \Vert},
```

where ``A(r)`` is the adhesion kernel:

```math
A(r) = \frac{0.007}{h_c^{3.25}}
\begin{cases}
\sqrt[4]{-\frac{4r^2}{h_c} + 6r - 2h_c}, & \text{if } 2r > h_c \text{ and } r \leq h_c, \\
0, & \text{otherwise.}
\end{cases}
```

---

### [Morris surface tension model](@id morris_csf)

The method described by [Morris](@cite Morris2000) estimates curvature by combining particle color gradients (see [`surface_normal`](@ref)) and smoothing functions to derive surface normals.
The computed curvature is then used to determine forces acting perpendicular to the interface.
While this method provides accurate surface tension forces, it does not explicitly conserve momentum.

In the Morris model, surface tension is computed based on local interface curvature ``\kappa`` and the unit surface normal ``\hat{\bm{n}}``.
By estimating ``\hat{\bm{n}}`` and ``\kappa`` at each particle near the interface, the
surface tension force for particle ``a`` can be written as

```math
\bm{F}_{a}^{\sigma}
= - m_a \sigma \frac{\kappa_a}{\rho_a}\hat{\bm{n}}_a.
```

This formulation focuses directly on geometric properties of the interface, making it relatively straightforward to implement when a reliable interface detection
(e.g., a color function) is available. However, accurately estimating ``\kappa`` and ``n`` may require fine resolutions.

---

### [Morris-based momentum-conserving surface tension model](@id moriss_css)

In addition to the simpler curvature-based formulation, [Morris](@cite Morris2000) introduced a momentum-conserving approach.
This method treats surface tension forces as arising from the divergence of a stress tensor, ensuring exact conservation
of linear momentum and offering more robust behavior for high-resolution or long-duration simulations
where accumulated numerical error can be significant.

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

This divergence can be computed numerically in the SPH framework as

```math
\bm{F}_{a}^{\sigma}
= m_a \sum_b \frac{m_b}{\rho_a \rho_b} (\bm{S}_a + \bm{S}_b) \nabla_a W_{ab}.
```

Implementation note: TrixiParticles.jl evaluates the corresponding acceleration and uses
the stabilized stress tensor
```math
\bm{S}_a^{\text{impl}}
= \delta_{s,a} (I - \hat{\bm{n}}_a \otimes \hat{\bm{n}}_a) - \delta_{s,\max} I,
```
with the factor ``\sigma`` applied outside the pairwise sum.

#### Advantages and limitations

While momentum conservation makes this model attractive, it requires additional computational effort and stabilization
techniques to address instabilities in high-density regions.

### API

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "surface_tension.jl")]
```

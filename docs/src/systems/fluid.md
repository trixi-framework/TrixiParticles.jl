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

The force exerted by particle ``b`` on particle ``a`` due to artificial viscosity is given by:

```math
F_{ab}^{\text{AV}} = - m_a m_b \Pi_{ab} \nabla W_{ab}
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

The term ``\mu_{ab}`` is defined as:

```math
\mu_{ab} = \frac{h \, v_{ab} \cdot r_{ab}}{\Vert r_{ab} \Vert^2 + \epsilon h^2},
```

with:

- ``h`` being the smoothing length,
- ``\epsilon`` a small parameter to prevent singularities,
- ``r_{ab} = r_a - r_b`` representing the difference of the coordinate vectors,
- ``v_{ab} = v_a - v_b`` representing the relative velocity between particles.

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

An additional force term ``\tilde{f}_{ab}`` is introduced to the pressure gradient force ``f_{ab}`` between particles ``a`` and ``b``:

```math
\tilde{f}_{ab} = m_a m_b \frac{(\mu_a + \mu_b)\, r_{ab} \cdot \nabla W_{ab}}{\rho_a \rho_b (\Vert r_{ab} \Vert^2 + \epsilon h^2)}\, v_{ab},
```

where:

- ``\mu_a = \rho_a \nu`` and ``\mu_b = \rho_b \nu`` represent the dynamic viscosities of particles ``a``and ``b`` (with ``\nu`` being the kinematic viscosity),
- ``r_{ab} = r_a - r_b`` represents the difference of the coordinate vectors,
- ``v_{ab} = v_a - v_b`` represents the relative velocity between particles.
- `` h `` is the smoothing length,
- `` \nabla W_{ab} `` is the gradient of the smoothing kernel,
- `` \epsilon `` is a small parameter to prevent singularities.

#### ViscosityAdami

`ViscosityAdami`, introduced by [Adami (2012)](@cite Adami2012), is optimized for incompressible or weakly compressible flows where precise modeling of shear stress is critical.
It enhances boundary layer representation by better resolving shear gradients, increasing dissipation in regions with steep velocity differences (e.g., near solid boundaries)
while minimizing compressibility effects. This results in accurate laminar flow simulations and accurate physical shear stresses.

##### Mathematical Formulation

The viscous interaction is modeled through a shear force for incompressible flows:

```math
f_{ab} = \sum_w \bar{\eta}_{ab} \left( V_a^2 + V_b^2 \right) \frac{v_{ab}}{||r_{ab}||^2 + \epsilon h_{ab}^2} \, (\nabla W_{ab} \cdot r_{ab}),
```

where:

- `` r_{ab} = r_a - r_b `` is the difference of the coordinate vectors,
- `` v_{ab} = v_a - v_b `` is their relative velocity,
- `` V_a = m_a / \rho_a`` and `` V_b = m_b / \rho_b`` are the particle volumes,
- `` h_{ab} `` is the smoothing length,
- `` \nabla W_{ab} `` is the gradient of the smoothing kernel,
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

### Configuration

Density and gradient corrections can be configured independently for WCSPH and EDAC:

```julia
fluid_system = WeaklyCompressibleSPHSystem(initial_condition;
                                           density_calculator=SummationDensity(),
                                           state_equation, smoothing_kernel,
                                           smoothing_length,
                                           density_correction=ShepardKernelCorrection(),
                                           gradient_correction=MixedKernelGradientCorrection())
```

The legacy `correction` keyword remains available for selecting one correction, but it cannot be
combined with the two role-specific keywords.

| System | Density handling | Supported correction behavior |
|:-------|:-----------------|:------------------------------|
| WCSPH with [`SummationDensity`](@ref) | Density is recomputed algebraically at every RHS evaluation | Shepard can filter density; all gradient corrections are supported |
| WCSPH with [`ContinuityDensity`](@ref) | Density is an evolved ODE variable | Gradient corrections are supported; Shepard is applied only by [`DensityReinitializationCallback`](@ref) |
| EDAC with [`SummationDensity`](@ref) | Density is algebraic and pressure is evolved independently | Same one-pass Shepard limitation as WCSPH; all gradient corrections are supported |
| EDAC with [`ContinuityDensity`](@ref) | Density and pressure are evolved ODE variables | Gradient corrections are supported; continuous Shepard density overwrite is rejected |
| [`ImplicitIncompressibleSPHSystem`](@ref) | Summation density is coupled to the pressure projection | Corrections are not supported |

IISPH relies on antisymmetric raw kernel gradients throughout its pressure matrix. Supporting an
asymmetric corrected gradient would require rederiving every projection term, so corrections are
intentionally not exposed for IISPH.

The default pressure acceleration is selected to match the density evolution law. When passing a
formulation explicitly, the caller is responsible for choosing the corresponding pairing:

| Pressure acceleration | Consistent density calculator | Asymmetric gradient corrections |
|:----------------------|:-------------------|:--------------------------------|
| `pressure_acceleration_summation_density` | [`SummationDensity`](@ref) | Supported |
| `pressure_acceleration_continuity_density` | [`ContinuityDensity`](@ref) | Supported |
| `inter_particle_averaged_pressure` | Either | Supported |
| [`tensile_instability_control`](@ref) | [`ContinuityDensity`](@ref) | Not supported |

For a correction whose gradients differ at particles ``a`` and ``b``, the conservative extensions
use both ``\widetilde{\nabla}W_{ab}^{(a)}`` and
``\widetilde{\nabla}W_{ba}^{(b)}``. They reduce algebraically to the original formulas when the
gradient is antisymmetric and give equal-and-opposite pair forces for arbitrary corrected
gradients. Tensile instability control has no such extension and is therefore rejected with
`KernelCorrection`, `GradientCorrection`, `BlendedGradientCorrection`, and
`MixedKernelGradientCorrection`.

### Consistency validation

The consistency degree of an SPH correction describes which polynomial fields its discrete
operator reproduces exactly. This is different from a convergence order: zeroth-order
consistency means exact constants, while first-order consistency means exact affine fields
([Bonet and Lok (1999)](@cite Bonet1999); [Sigalotti et al. (2021)](@cite Sigalotti2021)).

The correction operators are validated on regular and perturbed particle patches by comparing
their discrete moments with the analytical identities

```math
\sum_b V_b \widetilde{\nabla}W_{ab} = \bm{0}, \qquad
\sum_b V_b \widetilde{\nabla}W_{ab}(\bm{x}_b-\bm{x}_a)^T = \bm{I}.
```

The local truncation scaling follows by inserting a Taylor expansion into the discrete
interpolation ``I_h f`` and direct gradient ``G_h f``:

```math
\bm{M}_k = \sum_b V_b(\bm{x}_b-\bm{x}_a)^{\otimes k}W_{ab}, \qquad
\bm{G}_k = \sum_b V_b\widetilde{\nabla}W_{ab}
            \otimes(\bm{x}_b-\bm{x}_a)^{\otimes k}.
```

```math
I_h f-f_a = (M_0-1)f_a + \bm{M}_1 \cdot \nabla f_a
             + \frac{1}{2}\bm{M}_2 : \nabla^2 f_a + \cdots,
```

```math
G_h f-\nabla f_a = f_a\bm{G}_0 + (\bm{G}_1-\bm{I})\nabla f_a
                    + \frac{1}{2}\bm{G}_2 : \nabla^2 f_a + \cdots.
```

Here ``M_k=O(h^k)`` and ``G_k=O(h^{k-1})``. Consequently, exact constants give an
``O(h)`` interpolation on a generic one-sided support, while an exact linear gradient gives an
``O(h)`` derivative there. On a symmetric interior support, odd moments cancel and both can
display ``O(h^2)`` local truncation errors. The expected behavior of the implemented operators is:

| Correction | Enforced discrete moment | Generic/truncated support | Symmetric interior support |
|:-----------|:-------------------------|:--------------------------|:---------------------------|
| [`ShepardKernelCorrection`](@ref) | ``M_0=1`` | ``O(h)`` interpolation | ``O(h^2)`` interpolation |
| [`KernelCorrection`](@ref) | ``\bm{G}_0=\bm{0}`` | Removes the ``O(h^{-1})`` constant leakage, but leaves an ``O(1)`` first-moment error | No guaranteed rate at fixed ``\Delta x/h`` |
| [`GradientCorrection`](@ref) | ``\bm{G}_1=\bm{I}`` for the difference gradient | ``O(h)`` gradient | ``O(h^2)`` gradient |
| [`BlendedGradientCorrection`](@ref) | ``\bm{G}_1`` error scaled by ``1-\lambda`` | Fixed ``\lambda<1`` leaves an ``O(1)`` error | No guaranteed asymptotic rate at fixed ``\Delta x/h`` |
| [`MixedKernelGradientCorrection`](@ref) | ``\bm{G}_0=\bm{0}`` and ``\bm{G}_1=\bm{I}`` | ``O(h)`` gradient | ``O(h^2)`` gradient |

The Shepard interpolation scalings in this table assume prescribed volumes ``V_b`` that are
consistent with the interpolated field. The current [`SummationDensity`](@ref) implementation
instead forms ``V_b=m_b/\rho_b`` from the uncorrected summation density and performs one
normalization pass. At a truncated free surface with fixed ``\Delta x/h``, this reduces the error
constant but does not remove the ``O(1)`` boundary error. The validation therefore reports the
ideal normalized interpolation and the production summation-density update as separate operators.
The continuity-density reinitialization uses the evolved density as an independent volume source
and therefore recovers the expected Shepard scaling.

These are local operator scalings on self-similar regular particle patches with
``h\propto\Delta x``; they are not convergence rates of the complete SPH scheme. Classical SPH
also has a particle quadrature error depending on ``\Delta x/h`` and the particle distribution.
Formal convergence without consistency correction generally requires the joint limit
``h\to0``, ``\Delta x/h\to0``, and an increasing neighbor count
([Quinlan et al. (2006)](@cite Quinlan2006); [Zhu et al. (2015)](@cite Zhu2015)).

The reproducible study reports boundary and interior scalings separately. It prints a Markdown
table and writes its complete data to
`out/correction_convergence.csv`:

```bash
julia --project=. validation/corrections/convergence.jl
```

#### Measured operator scaling

The following values are from the finest refinement (``N=96`` particles per coordinate
direction) of a cubic manufactured field with a [`WendlandC6Kernel`](@ref),
``h/\Delta x=2``, and prescribed particle volumes.
The boundary sample uses a one-sided kernel support away from the corners; the interior sample
has a complete symmetric support. The measured scaling is calculated between ``N=48`` and
``N=96``. The summation-density rows use the production [`SummationDensity`](@ref) update instead
of prescribed volumes; the reinitialization row uses the evolved continuity density as its volume
source.

| Method | Operator | Boundary ``L_2`` error | Boundary scaling | Interior ``L_2`` error | Interior scaling |
|:-------|:---------|-----------------------:|-----------------:|-----------------------:|-----------------:|
| Uncorrected | Interpolation | ``2.622e-1`` | ``-0.007`` | ``2.956e-4`` | ``0.543`` |
| [`ShepardKernelCorrection`](@ref) | Normalized interpolation | ``1.567e-3`` | ``1.052`` | ``4.466e-5`` | ``2.014`` |
| Uncorrected | Direct gradient | ``6.339e1`` | ``-1.009`` | ``7.226e-4`` | ``-0.090`` |
| [`KernelCorrection`](@ref) | Direct gradient | ``3.200e-1`` | ``-0.005`` | ``9.735e-4`` | ``-0.068`` |
| [`GradientCorrection`](@ref) | Difference gradient | ``1.123e-2`` | ``1.010`` | ``2.381e-5`` | ``2.016`` |
| [`BlendedGradientCorrection`](@ref), ``\lambda=0.5`` | Difference gradient | ``1.765e-1`` | ``-0.049`` | ``3.539e-4`` | ``-0.170`` |
| [`MixedKernelGradientCorrection`](@ref) | Direct gradient | ``9.173e-3`` | ``1.012`` | ``2.381e-5`` | ``2.016`` |
| Uncorrected | Summation density | ``2.631e-1`` | ``-0.002`` | ``2.537e-4`` | ``0.040`` |
| [`ShepardKernelCorrection`](@ref) | Summation density | ``1.918e-1`` | ``-0.004`` | ``2.557e-4`` | ``0.076`` |
| [`ShepardKernelCorrection`](@ref) | Continuity-density reinitialization | ``3.810e-4`` | ``1.010`` | ``2.401e-6`` | ``2.001`` |

#### Measured pressure acceleration

The same study evaluates every supported pressure-acceleration pairing with a manufactured
pressure that vanishes at the left free surface and the exact acceleration
``-\nabla p/\rho``. Since a conservative asymmetric pair uses correction data from both particles,
the interior sample keeps both kernel neighborhoods complete. Each entry below is the interior
``L_2`` error at ``N=96`` followed by the scaling from ``N=48`` to ``N=96``.

| Correction | Summation-density pressure | Inter-particle, summation density | Continuity-density pressure | Inter-particle, continuity density |
|:-----------|----------------------------:|----------------------------------:|----------------------------:|----------------------------------:|
| None | ``9.557e-4 / -0.162`` | ``9.557e-4 / -0.162`` | ``7.046e-4 / -0.224`` | ``7.046e-4 / -0.224`` |
| [`ShepardKernelCorrection`](@ref) | ``9.557e-4 / -0.162`` | ``9.557e-4 / -0.162`` | Not applicable | Not applicable |
| [`KernelCorrection`](@ref) | ``9.557e-4 / -0.162`` | ``9.557e-4 / -0.162`` | ``9.557e-4 / -0.162`` | ``9.557e-4 / -0.162`` |
| [`GradientCorrection`](@ref) | ``3.439e-5 / 2.021`` | ``3.439e-5 / 2.021`` | ``3.439e-5 / 2.021`` | ``3.439e-5 / 2.021`` |
| [`BlendedGradientCorrection`](@ref), ``\lambda=0.5`` | ``4.613e-4 / -0.355`` | ``4.613e-4 / -0.355`` | ``3.358e-4 / -0.509`` | ``3.358e-4 / -0.509`` |
| [`MixedKernelGradientCorrection`](@ref) | ``3.439e-5 / 2.021`` | ``3.439e-5 / 2.021`` | ``3.439e-5 / 2.021`` | ``3.439e-5 / 2.021`` |
| Shepard density + mixed gradient | ``3.465e-5 / 2.004`` | ``3.465e-5 / 2.005`` | Not applicable | Not applicable |

For positive pressure, [`tensile_instability_control`](@ref) reduces exactly to the uncorrected
continuity-density pressure law and produces the same measured errors. All formulations have a
constant-pressure null response in the complete interior to an absolute acceleration error below
``1e-7``.

On the truncated free-surface row, none of the conservative pressure operators converges at fixed
``h/\Delta x``; the observed scaling remains approximately zero. The local correction moments
constrain one particle's gradient operator, but do not impose consistency on a conservative pair
assembled from two differently truncated neighborhoods. This limitation is reported rather than
hidden by applying a non-conservative one-sided pressure difference. The complete boundary and
interior data for every variation are written to `out/correction_convergence.csv`.

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

The simplest SPH formulation for a surface normal, ``n_a`` is given as

```math
n_a = \sum_b m_b \frac{c_b}{\rho_b} \nabla_a W_{ab},
```

where:

- ``c_b`` is the color field value for particle ``b``,
- ``m_b`` is the mass of particle ``b``,
- ``\rho_b`` is the density of particle ``b``,
- ``\nabla_a W_{ab}`` is the gradient of the smoothing kernel ``W_{ab}`` with respect to particle ``a``.

#### Normalization of surface normals

The calculated normals are normalized to unit vectors:

```math
\hat{n}_a = \frac{n_a}{\Vert n_a \Vert}.
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

Mathematically:

```math
F_{\text{cohesion}} = -\sigma m_b C(r) \frac{r}{\Vert r \Vert},
```

where ``C(r)``, the cohesion kernel, is defined as:

```math
C(r)=\frac{32}{\pi h_c^9}
\begin{cases}
(h_c-r)^3 r^3, & \text{if } 2r > h_c, \\
2(h_c-r)^3 r^3 - \frac{h^6}{64}, & \text{if } r > 0 \text{ and } 2r \leq h_c, \\
0, & \text{otherwise.}
\end{cases}
```

#### Surface area minimization force

The surface area minimization force models the curvature reduction effects, aligning particle motion to reduce the interface's total area.
It acts based on the difference in surface normals:

```math
F_{\text{curvature}} = -\sigma (n_a - n_b),
```

where ``n_a`` and ``n_b`` are the surface normals of the interacting particles.

#### Wall adhesion force

This force models the interaction between fluid and solid boundaries, simulating adhesion effects at walls.
It uses a custom kernel with a peak at 0.75 times the support radius:

```math
F_{\text{adhesion}} = -\beta m_b A(r) \frac{r}{\Vert r \Vert},
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

In the Morris model, surface tension is computed based on local interface curvature ``\kappa`` and the unit surface normal ``\hat{n}.``
By estimating ``\hat{n}`` and ``\kappa`` at each particle near the interface, the surface tension force for particle a can be written as:

```math
F_{\text{surface tension}} = - \sigma \frac{\kappa_a}{\rho_a}\hat{n}_a
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

The surface tension force can be seen as a divergence of a stress tensor ``S``

```math
F_{\text{surface tension}} = \nabla \cdot S,
```

with ``S`` defined as

```math
S = \sigma \delta_s (I - \hat{n} \otimes \hat{n}),
```

with:

- ``\delta_s``: Surface delta function,
- ``\hat{n}``: Unit normal vector,
- ``I``: Identity matrix.

This divergence can be computed numerically in the SPH framework as

```math
\sum_b \frac{m_b}{\rho_a \rho_b} (S_a + S_b) \nabla W_{ab}
```

#### Advantages and limitations

While momentum conservation makes this model attractive, it requires additional computational effort and stabilization
techniques to address instabilities in high-density regions.

### API

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "surface_tension.jl")]
```

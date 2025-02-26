
# [Fluid Models](@id fluid_models)

Currently available fluid methods are the [weakly compressible SPH method](@ref wcsph) and the
[entropically damped artificial compressibility for SPH](@ref edac).  
This page lists models and techniques that apply to both of these methods.  

## [Viscosity](@id viscosity_wcsph)

Viscosity is a critical physical property governing momentum diffusion within a fluid.
In the context of SPH, viscosity determines how rapidly velocity gradients are smoothed out,
influencing key flow characteristics such as boundary layer formation, vorticity diffusion,
and dissipation of kinetic energy. It also helps determine whether a flow is laminar or turbulent
under a given set of conditions.

Implementing viscosity correctly in SPH is essential for producing physically accurate results,
and different methods exist to capture both numerical stabilization and true viscous effects.

### API

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
direct experimental measurements or specialized equations of state may be necessary.

| **Fluid**    | **Surface Tension (``\sigma``) [N/m at 20°C]** |
|--------------|----------------------------------------------:|
| **Water**    | 0.0728                                        |
| **Mercury**  | 0.485                                         |
| **Ethanol**  | 0.0221                                        |
| **Acetone**  | 0.0237                                        |
| **Glycerol** | 0.0634                                        |
| **Olive Oil**| ~0.032                                        |
| **Gasoline** | ~0.022                                        |
| **Mineral Oil** | ~0.030                                     |

### [Akinci-based intra-particle force surface tension and wall adhesion model](@id akinci_ipf)

The Akinci model divides surface tension into distinct force components:

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

The method estimates curvature by combining particle color gradients (see [`surface_normal`](@ref)) and smoothing functions to derive surface normals.
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

In addition to the simpler curvature-based formulation, Morris (2000) introduced a momentum-conserving approach.
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


# [Huber Model](@id huber_model)

## Introduction to the Huber Contact Force Model

The **Huber Model**, introduced in [Huber2016](@cite), provides a physically-based approach for simulating surface tension and contact line dynamics in Smoothed Particle Hydrodynamics (SPH). It is specifically designed to address challenges in modeling wetting phenomena, including dynamic contact angles and their influence on fluid behavior.

The Huber Model introduces a **Contact Line Force (CLF)** that complements the **Continuum Surface Force (CSF)** model, allowing for accurate representation of fluid-fluid and fluid-solid interactions. The dynamic evolution of contact angles emerges naturally from the force balance without requiring artificial adjustments or fitting parameters.

---

## Key Features of the Huber Model

1. **Dynamic Contact Angles**:
   - Captures the transition between static and dynamic contact angles based on interfacial forces and contact line velocities.
   - Removes the need for predefined constitutive equations, as the contact angle is derived from the system's dynamics.

2. **Volume Reformulation**:
   - Transforms line-based forces (e.g., those acting at the contact line) into volume-based forces for compatibility with SPH formulations.
   - Ensures smooth force distribution near the contact line, reducing numerical artifacts.

3. **Momentum Balance**:
   - Extends the Navier-Stokes equations to include contributions from the contact line, ensuring accurate modeling of wetting and spreading dynamics.

4. **No Fitting Parameters**:
   - Fully physics-driven, requiring only measurable inputs like surface tension coefficients and static contact angles.

---

## Mathematical Formulation

### Contact Line Force (CLF)

The force acting along the contact line is derived from the unbalanced Young Force:
```math
f_{\text{CLF}} = \sigma_{\text{wn}} [\cos(\alpha_s) - \cos(\alpha_d)] \hat{\nu},
```
where:
- \( \sigma_{\text{wn}} \): Surface tension coefficient of the fluid-fluid interface,
- \( \alpha_s \): Static contact angle,
- \( \alpha_d \): Dynamic contact angle,
- \( \hat{\nu} \): Tangential unit vector along the fluid-solid interface.

### Volume Reformulation

To incorporate the CLF into SPH, it is reformulated as a volume force:
```math
F_{\text{CLF}} = f_{\text{CLF}} \delta_{\text{CL}},
```
where \( \delta_{\text{CL}} \) is a Dirac delta function approximated by SPH kernels, ensuring the force is applied locally near the contact line.

---

## Applications

1. **Droplet Dynamics**:
   - Simulates droplet spreading, recoiling, and merging with accurate contact line evolution.

2. **Capillary Action**:
   - Models fluid behavior in porous media and confined geometries, where contact lines play a critical role.

3. **Wetting Phenomena**:
   - Predicts equilibrium shapes and transient states of droplets and films on solid surfaces.

# [Huber Model](@id huber_model)

## Introduction to the Huber Contact Force Model

The **Huber Model**, introduced in [Huber2016](@cite), provides a physically-based approach for simulating surface tension and contact line dynamics in Smoothed Particle Hydrodynamics (SPH). It is specifically designed to address challenges in modeling wetting phenomena, including dynamic contact angles and their influence on fluid behavior.

The Huber Model introduces a **Contact Line Force (CLF)** that complements the **Continuum Surface Force (CSF)** model, allowing for accurate representation of fluid-fluid and fluid-solid interactions. The dynamic evolution of contact angles emerges naturally from the force balance without requiring artificial adjustments or fitting parameters.

---

## Key Features of the Huber Model

1. **Dynamic Contact Angles**:
   - Captures the transition between static and dynamic contact angles based on interfacial forces and contact line velocities.
   - Removes the need for predefined constitutive equations, as the contact angle is derived from the system's dynamics.

2. **Volume Reformulation**:
   - Transforms line-based forces (e.g., those acting at the contact line) into volume-based forces for compatibility with SPH formulations.
   - Ensures smooth force distribution near the contact line, reducing numerical artifacts.

3. **Momentum Balance**:
   - Extends the Navier-Stokes equations to include contributions from the contact line, ensuring accurate modeling of wetting and spreading dynamics.

4. **No Fitting Parameters**:
   - Fully physics-driven, requiring only measurable inputs like surface tension coefficients and static contact angles.

---

## Mathematical Formulation

### Contact Line Force (CLF)

The force acting along the contact line is derived from the unbalanced Young Force:
```math
f_{\text{CLF}} = \sigma_{\text{wn}} [\cos(\alpha_s) - \cos(\alpha_d)] \hat{\nu},
```
where:
- \( \sigma_{\text{wn}} \): Surface tension coefficient of the fluid-fluid interface,
- \( \alpha_s \): Static contact angle,
- \( \alpha_d \): Dynamic contact angle,
- \( \hat{\nu} \): Tangential unit vector along the fluid-solid interface.

### Volume Reformulation

To incorporate the CLF into SPH, it is reformulated as a volume force:
```math
F_{\text{CLF}} = f_{\text{CLF}} \delta_{\text{CL}},
```
where \( \delta_{\text{CL}} \) is a Dirac delta function approximated by SPH kernels, ensuring the force is applied locally near the contact line.

---

## Applications

1. **Droplet Dynamics**:
   - Simulates droplet spreading, recoiling, and merging with accurate contact line evolution.

2. **Capillary Action**:
   - Models fluid behavior in porous media and confined geometries, where contact lines play a critical role.

3. **Wetting Phenomena**:
   - Predicts equilibrium shapes and transient states of droplets and films on solid surfaces.

### API

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "surface_tension.jl")]
```

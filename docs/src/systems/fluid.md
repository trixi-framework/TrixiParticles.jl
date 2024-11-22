
# [Fluid Models](@id fluid_models)

Below are common models for fluid effects used by both EDAC and WCSPH.

---

## [Surface Normals](@id surface_normal)

### Overview of Surface Normal Calculation in SPH

Surface normals are essential for modeling surface tension as they provide the directionality of forces acting at the fluid interface. They are calculated based on the particle properties and their spatial distribution within the smoothed particle hydrodynamics (SPH) framework.

#### Color Field and Gradient-Based Surface Normals

The surface normal at a particle is derived from the color field, a scalar field assigned to particles to distinguish between different fluid phases or between fluid and air. The color field gradients point towards the interface, and the normalized gradient defines the surface normal direction.

The simplest SPH formulation for surface normal, \( n_a \), is given as:
```math
n_a = \sum_b m_b \frac{c_b}{\rho_b} \nabla_a W_{ab},
```
where:
- \( c_b \) is the color field value for particle \( b \),
- \( m_b \) is the mass of particle \( b \),
- \( \rho_b \) is the density of particle \( b \),
- \( \nabla_a W_{ab} \) is the gradient of the smoothing kernel \( W_{ab} \) with respect to particle \( a \).

#### Normalization of Surface Normals

The calculated normals are normalized to unit vectors:
```math
\hat{n}_a = \frac{n_a}{\Vert n_a \Vert}.
```
Normalization ensures that the magnitude of the normals does not bias the curvature calculations or the resulting surface tension forces.

#### Handling Noise and Errors in Normal Calculation

In regions distant from the interface, the calculated normals may be small or inaccurate due to the smoothing kernel's support radius. To mitigate this:
1. Normals below a threshold are excluded from further calculations.
2. Curvature calculations use a corrected formulation to reduce errors near interface fringes.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "surface_normal_sph.jl")]
```

---

## [Surface Tension](@id surface_tension)

### Introduction to Surface Tension in SPH

Surface tension is a key phenomenon in fluid dynamics, influencing the behavior of droplets, bubbles, and fluid interfaces. In SPH, surface tension is modeled as forces arising due to surface curvature and particle interactions, ensuring realistic simulation of capillary effects, droplet coalescence, and fragmentation.

### Akinci-Based Intra-Particle Force Surface Tension and Wall Adhesion Model

The Akinci model divides surface tension into distinct force components:

#### Cohesion Force

The cohesion force captures the attraction between particles at the fluid interface, creating the effect of surface tension. It is defined by the distance between particles and the support radius \( h_c \), using a kernel-based formulation.

**Key Features:**
- Particles within half the support radius experience a repulsive force to prevent clustering.
- Particles beyond half the radius but within the support radius experience an attractive force to simulate cohesion.

Mathematically:
```math
F_{\text{cohesion}} = -\sigma m_b C(r) \frac{r}{\Vert r \Vert},
```
where \( C(r) \), the cohesion kernel, is defined as:
```math
C(r)=\frac{32}{\pi h_c^9}
\begin{cases}
(h_c-r)^3 r^3, & \text{if } 2r > h_c, \\
2(h_c-r)^3 r^3 - \frac{h^6}{64}, & \text{if } r > 0 \text{ and } 2r \leq h_c, \\
0, & \text{otherwise.}
\end{cases}
```

#### Surface Area Minimization Force

The surface area minimization force models the curvature reduction effects, aligning particle motion to reduce the interface's total area. It acts based on the difference in surface normals:
```math
F_{\text{curvature}} = -\sigma (n_a - n_b),
```
where \( n_a \) and \( n_b \) are the surface normals of the interacting particles.

#### Wall Adhesion Force

This force models the interaction between fluid and solid boundaries, simulating adhesion effects at walls. It uses a custom kernel with a peak at 0.75 times the support radius:
```math
F_{\text{adhesion}} = -\beta m_b A(r) \frac{r}{\Vert r \Vert},
```
where \( A(r) \) is the adhesion kernel:
```math
A(r) = \frac{0.007}{h_c^{3.25}}
\begin{cases}
\sqrt[4]{-\frac{4r^2}{h_c} + 6r - 2h_c}, & \text{if } 2r > h_c \text{ and } r \leq h_c, \\
0, & \text{otherwise.}
\end{cases}
```

---

### Morris-Based Momentum-Conserving Surface Tension Model

In addition to the Akinci model, Morris (2000) introduced a momentum-conserving approach to surface tension. This model uses stress tensors to ensure exact conservation of linear momentum, providing a robust method for high-resolution simulations.

#### Stress Tensor Formulation

The force is calculated as:
```math
F_{\text{surface tension}} = \nabla \cdot S,
```
where \( S \) is the stress tensor:
```math
S = \sigma \delta_s (I - \hat{n} \otimes \hat{n}),
```
with:
- \( \delta_s \): Surface delta function,
- \( \hat{n} \): Unit normal vector,
- \( I \): Identity matrix.

#### Advantages and Limitations

While momentum conservation makes this model attractive, it requires additional computational effort and stabilization techniques to address instabilities in high-density regions.

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

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "surface_tension.jl")]
```


# [Fluid Models](@id fluid_models)
Currently available fluid methods are the [weakly compressible SPH method](@ref wcsph) and the [entropically damped artificial compressibility for SPH](@ref edac).  
This page lists models and techniques that apply to both of these methods.  

## [Viscosity](@id viscosity_wcsph)

TODO: Explain viscosity.

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

Surface normals are essential for modeling surface tension as they provide the directionality of forces acting at the fluid interface. They are calculated based on the particle properties and their spatial distribution within the smoothed particle hydrodynamics (SPH) framework.

#### Color field and gradient-based surface normals

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

#### Normalization of surface normals

The calculated normals are normalized to unit vectors:
```math
\hat{n}_a = \frac{n_a}{\Vert n_a \Vert}.
```
Normalization ensures that the magnitude of the normals does not bias the curvature calculations or the resulting surface tension forces.

#### Handling noise and errors in normal calculation

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

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "surface_tension.jl")]
```

This extended documentation provides a comprehensive view of the theoretical foundations and practical implementations of surface tension and surface normal calculations in SPH models.

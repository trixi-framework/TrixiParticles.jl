
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

### Artificial (numerical) viscosity

- Goal: Stabilize the simulation, capture shocks, and prevent unphysical particle interpenetration.
- Method: Adds a dissipative (artificial) term to the momentum equations.
- Typical Use: High-speed flows with strong shocks, astrophysical simulations, or situations where numerical damping is needed for stability.

### Physical (real) viscosity

- Goal: Model the actual viscous stresses of a fluid, aligned with a target Reynolds number or experimentally measured fluid properties.
- Method: Introduces a force consistent with the Navier–Stokes viscous stress term.
- Typical Use: Low-speed, incompressible or weakly compressible flows where matching real fluid behavior is important.

### Model comparison

#### ArtificialViscosityMonaghan

- Best For: Compressible/high-speed flows, shock capturing, general purpose damping.
- If you need: Stability in challenging flow regimes with potentially large density/pressure variations.

#### ViscosityMorris

- Best For: Moderate to low Mach number flows where realistic viscous behavior is desired.
- If you need: Straightforward approach to physical viscosity that still works well in weakly compressible scenarios.

#### ViscosityAdami

- Best For: Incompressible or weakly compressible flows requiring accurate shear stress treatment.
- If you need: Good boundary layer representation and accurate laminar flow with minimal compressibility effects.


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

Surface normals are essential for modeling surface tension as they provide the directionalit
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
In SPH, surface tension is modeled as forces arising due to surface curvature and particle interactions, ensuring realistic
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


### Akinci-Based Intra-Particle Force Surface Tension and Wall Adhesion Model

The Akinci model divides surface tension into distinct force components:

#### Cohesion Force

The cohesion force captures the attraction between particles at the fluid interface, creating the effect of surface tension.
It is defined by the distance between particles and the support radius ``h_c``, using a kernel-based formulation.

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

The surface area minimization force models the curvature reduction effects, aligning particle motion to reduce the interface's total area.
It acts based on the difference in surface normals:
```math
F_{\text{curvature}} = -\sigma (n_a - n_b),
```
where \( n_a \) and \( n_b \) are the surface normals of the interacting particles.

#### Wall Adhesion Force

This force models the interaction between fluid and solid boundaries, simulating adhesion effects at walls.
It uses a custom kernel with a peak at 0.75 times the support radius:
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

### Morris Surface Tension Model

The method estimates curvature by combining particle color gradients and smoothing functions to derive surface normals.
The computed curvature is then used to determine forces acting perpendicular to the interface.
While this method provides accurate surface tension forces, it does not explicitly conserve momentum.

In the Morris model, surface tension is computed based on local interface curvature ``\kappa`` and the unit surface normal ``\hat{n}``.
By estimating ``\hat{n}`` and ``\kappa`` at each particle near the interface, the surface tension force for particle a can be written as:

```math
F_{\text{surface tension}} = - \sigma \frac{\kappa_a}{\rho_a}\hat{n}_a
```
This formulation focuses directly on geometric properties of the interface, making it relatively straightforward to implement when a reliable interface detection 
(e.g., a color function) is available. However, accurately estimating ``\kappa`` and ``n`` may require fine resolutions.
---

### Morris-Based Momentum-Conserving Surface Tension Model

In addition to the simpler curvature-based formulation, Morris (2000) introduced a momentum-conserving approach.
This method treats surface tension forces as arising from the divergence of a stress tensor, ensuring exact conservation
of linear momentum and offering more robust behavior for high-resolution or long-duration simulations
where accumulated numerical error can be significant.

#### Stress Tensor Formulation

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

That can be calculated as
```math 
\sum_b \frac{m_b}{\rho_a \rho_b} (S_a + S_b) \nabla W_{ab} 
```

#### Advantages and Limitations

While momentum conservation makes this model attractive, it requires additional computational effort and stabilization
techniques to address instabilities in high-density regions.

### API

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "surface_tension.jl")]
```
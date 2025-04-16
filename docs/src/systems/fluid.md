# [Fluid Models](@id fluid_models)

Currently available fluid methods are the [weakly compressible SPH method](@ref wcsph) and the [entropically damped artificial compressibility for SPH](@ref edac).  
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

Artificial (numerical) viscosity is a technique used to stabilize simulations of inviscid flows, which would otherwise show unphysical particle movement due to numerical instability.
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
is designed primarily for compressible,
high-speed flows where shock capturing is critical.
In its implementation, the method includes a dissipation term
that increases when particles approach each other.
This increase in dissipation is triggered by the relative motion between particles:
as particles come closer and compress the local flow,
the artificial viscosity term becomes stronger to damp out rapid changes
and prevent unphysical clustering.
This ensures that while the simulation remains stable in challenging
flow regimes with large density or pressure variations,
the physical behavior is not overly altered.

##### Mathematical Formulation

The artificial viscosity between two particles ``a`` and ``b`` is given by:

```math
\Pi_{ab} =
\begin{cases}
    -\frac{\alpha c \mu_{ab} + \beta \mu_{ab}^2}{\bar{\rho}_{ab}} & \text{if } v_{ab} \cdot r_{ab} < 0, \\
    0 & \text{otherwise}
\end{cases}
```

where:

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

To ensure that the simulation maintains a consistent Reynolds number when the resolution changes, the parameter ``\alpha`` must be adjusted accordingly. Monaghan (2005) introduced an effective physical kinematic viscosity ``\nu`` defined as:

```math
\nu = \frac{\alpha h c}{2d + 4},
```

where **``d``** is the number of spatial dimensions. This relation allows the calibration of ``\alpha`` to achieve the desired viscous behavior as the resolution or simulation conditions vary.

#### ViscosityMorris

`ViscosityMorris` is ideal for moderate to low Mach number flows where accurately modeling physical viscous behavior is essential. Developed by [Morris (1997)](@cite Morris1997) and later applied by [Fourtakas (2019)](@cite Fourtakas2019), this method directly simulates the viscous stresses found in fluids rather than relying on artificial viscosity.

By approximating momentum diffusion based on local fluid properties, the method captures the actual viscous forces without excessive damping. This results in a more realistic representation of flow dynamics in weakly compressible scenarios.

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

ViscosityAdami, introduced by [Adami (2012)](@cite Adami2012), is optimized for incompressible or weakly compressible flows where precise modeling of shear stress is critical. It enhances boundary layer representation by better resolving shear gradients, increasing dissipation in regions with steep velocity differences (e.g., near solid boundaries) while minimizing compressibility effects. This results in accurate laminar flow simulations and a faithful depiction of physical shear stresses.

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

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "viscosity.jl")]
```

## [Corrections](@id corrections)

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("general", "corrections.jl")]
```



## [Surface Normals](@id surface_normal)
```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "surface_normal_sph.jl")]
```

## [Surface Tension](@id surface_tension)

### Akinci-based intra-particle force surface tension and wall adhesion model
The work by Akinci proposes three forces:
- a cohesion force
- a surface area minimization force
- a wall adhesion force

The classical model is composed of the curvature minimization and cohesion force.

#### Cohesion force
The model calculates the cohesion force based on the support radius ``h_c`` and the distance between particles.
This force is determined using two distinct regimes within the support radius:
- For particles closer than half the support radius,
  a repulsive force is calculated to prevent particles from clustering too tightly,
  enhancing the simulation's stability and realism.
- Beyond half the support radius and within the full support radius,
  an attractive force is computed, simulating the effects of surface tension that draw particles together.
The cohesion force, ``F_{\text{cohesion}}``, for a pair of particles is given by:
```math
F_{\text{cohesion}} = -\sigma m_b C(r) \frac{r}{\Vert r \Vert},
```
where:
- ``\sigma`` represents the surface tension coefficient, adjusting the overall strength of the cohesion effect.
- ``C`` is a scalar function of the distance between particles.

The cohesion kernel ``C`` is defined as
```math
C(r)=\frac{32}{\pi h_c^9}
\begin{cases}
(h_c-r)^3 r^3, & \text{if } 2r > h_c \\
2(h_c-r)^3 r^3 - \frac{h^6}{64}, & \text{if } r > 0 \text{ and } 2r \leq h_c \\
0, & \text{otherwise}
\end{cases}
```

#### Surface area minimization force
To model the minimization of the surface area and curvature of the fluid, a curvature force is used, which is calculated as
```math
F_{\text{curvature}} = -\sigma (n_a - n_b)
```

#### Wall adhesion force
The wall adhesion model proposed by Akinci et al. is based on a kernel function which is 0 from 0.0 to 0.5 support radiia with a maximum at 0.75.
With the force calculated with an adhesion coefficient ``\beta`` as
```math
F_{\text{adhesion}} = -\beta m_b A(r) \frac{r}{\Vert r \Vert},
```
with ``A`` being the adhesion kernel defined as
```math
A(r)= \frac{0.007}{h_c^{3.25}}
\begin{cases}
\sqrt[4]{- \frac{4r^2}{h_c} + 6r - 2h_c}, & \text{if } 2r > h_c \text{ and } r \leq h_c \\
0, & \text{otherwise.}
\end{cases}
```

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "surface_tension.jl")]
```

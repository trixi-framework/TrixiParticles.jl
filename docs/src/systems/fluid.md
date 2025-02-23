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

Artificial (numerical) viscosity is a technique used to stabilize simulations of inviscid flows by preventing
unphysical artificial overlap or penetration of particles that would not occur in a real fluid.
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

ArtificialViscosityMonaghan is designed primarily for compressible, high-speed flows where shock capturing is critical.
In its implementation, the method includes a dissipation term that increases when particles approach each other.
This increase in dissipation is triggered by the relative motion between particles: as particles come closer and compress the local flow,
the artificial viscosity term becomes stronger to damp out rapid changes and prevent unphysical clustering.
This ensures that while the simulation remains stable in challenging flow regimes with large density or pressure variations,
the physical behavior is not overly altered.

#### ViscosityMorris

ViscosityMorris is well-suited for moderate to low Mach number flows where modeling realistic viscous behavior is important.
Unlike artificial viscosity, this approach directly simulates the physical viscous stresses encountered in fluids.
Its implementation approximates the diffusion of momentum in a way that naturally reflects the viscous behavior observed in experiments
or specified by a target Reynolds number. Because it mimics the actual viscous forces without introducing excessive damping,
it works effectively in weakly compressible scenarios, allowing for a more accurate representation of the flow dynamics.

#### ViscosityAdami

ViscosityAdami is optimized for incompressible or weakly compressible flows, particularly where an accurate treatment of shear stress is needed.
The method improves the representation of boundary layers by using a refined approach that better resolves shear gradients.
In practice, this means that the method enhances the dissipation in regions with steep velocity differences (such as near solid boundaries)
while keeping compressibility effects minimal. As a result, it delivers accurate laminar flow simulations and a more faithful depiction of the physical shear stress,
which is essential for problems where boundary layer behavior plays a crucial role.

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

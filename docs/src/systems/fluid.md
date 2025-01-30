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

# [Weakly Compressible SPH](@id wcsph)

Weakly compressible SPH as introduced by [Monaghan (1994)](@cite Monaghan1994). This formulation relies on a stiff
[equation of state](@ref equation_of_state) that generates large pressure changes
for small density variations.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "weakly_compressible_sph", "system.jl")]
```

## [Equation of State](@id equation_of_state)

The equation of state is used to relate fluid density to pressure and thus allow
an explicit simulation of the [WCSPH system](@ref WeaklyCompressibleSPHSystem).
The equation in the following formulation was introduced by [Cole (1948)](@cite Cole1948) (pp. 39 and 43).
The pressure ``p`` is calculated as
```math
    p = B \left(\left(\frac{\rho}{\rho_0}\right)^\gamma - 1\right) + p_{\text{background}},
```
where ``\rho`` denotes the density, ``\rho_0`` the reference density,
and ``p_{\text{background}}`` the background pressure, which is set to zero when applied to
free-surface flows ([Adami et al., 2012](@cite Adami2012)).

The bulk modulus, ``B =  \frac{\rho_0 c^2}{\gamma}``, is calculated from the artificial
speed of sound ``c`` and the isentropic exponent ``\gamma``.

An ideal gas equation of state with a linear relationship between pressure and density can
be obtained by choosing `exponent=1`, i.e.
```math
    p = B \left( \frac{\rho}{\rho_0} -1 \right) = c^2(\rho - \rho_0).
```

For higher Reynolds numbers, `exponent=7` is recommended, whereas at lower Reynolds
numbers `exponent=1` yields more accurate pressure estimates since pressure and
density are proportional (see [Morris, 1997](@cite Morris1997)).

When using [`SummationDensity`](@ref) (or [`DensityReinitializationCallback`](@ref))
and free surfaces, initializing particles with equal spacing will cause underestimated
density and therefore strong attractive forces between particles at the free surface.
Setting `clip_negative_pressure=true` can avoid this.
```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "weakly_compressible_sph", "state_equations.jl")]
```

## [Viscosity](@id viscosity_wcsph)

TODO: Explain viscosity.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "viscosity.jl")]
```

## [Density Diffusion](@id density_diffusion)

Density diffusion can be used with [`ContinuityDensity`](@ref) to remove the noise in the
pressure field. It is highly recommended to use density diffusion when using WCSPH.

### Formulation

All density diffusion terms extend the continuity equation (see [`ContinuityDensity`](@ref))
by an additional term
```math
\frac{\mathrm{d}\rho_a}{\mathrm{d}t} = \sum_{b} m_b v_{ab} \cdot \nabla_{r_a} W(\Vert r_{ab} \Vert, h)
    + \delta h c \sum_{b} V_b \psi_{ab} \cdot \nabla_{r_a} W(\Vert r_{ab} \Vert, h),
```
where ``V_b = m_b / \rho_b`` is the volume of particle ``b`` and ``\psi_{ab}`` depends on
the density diffusion method (see [`DensityDiffusion`](@ref) for available terms).
Also, ``\rho_a`` denotes the density of particle ``a`` and ``r_{ab} = r_a - r_b`` is the
difference of the coordinates, ``v_{ab} = v_a - v_b`` of the velocities of particles
``a`` and ``b``.

### Numerical Results

All density diffusion terms remove numerical noise in the pressure field and produce more
accurate results than weakly commpressible SPH without density diffusion.
This can be demonstrated with dam break examples in 2D and 3D. Here, ``δ = 0.1`` has
been used for all terms.
Note that, due to added stability, the adaptive time integration method that was used here
can choose higher time steps in the simulations with density diffusion.
For the cheap [`DensityDiffusionMolteniColagrossi`](@ref), this results in reduced runtime.

```@raw html
<figure>
  <img src="https://github.com/trixi-framework/TrixiParticles.jl/assets/44124897/01289e3b-98ce-4b2d-8151-cd20782d5823" alt="density_diffusion_2d"/>
  <figcaption>Dam break in 2D with different density diffusion terms</figcaption>
</figure>
```

```@raw html
<figure>
  <img src="https://github.com/trixi-framework/TrixiParticles.jl/assets/44124897/63a05b2a-6c37-468e-b895-15ab142a4eba" alt="density_diffusion_3d"/>
  <figcaption>Dam break in 3D with different density diffusion terms</figcaption>
</figure>
```

The simpler terms [`DensityDiffusionMolteniColagrossi`](@ref) and
[`DensityDiffusionFerrari`](@ref) do not solve the hydrostatic problem and lead to incorrect
solutions in long-running steady-state hydrostatic simulations with free surfaces
[(Antuono et al., 2012)](@cite Antuono2012). This can be seen when running the simple rectangular tank example
until ``t = 40`` (again using ``δ = 0.1``):

```@raw html
<figure>
  <img src="https://github.com/trixi-framework/TrixiParticles.jl/assets/44124897/440debc9-6051-4a3b-aa9c-02a6b32fccf3" alt="density_diffusion_tank"/>
  <figcaption>Tank in rest under gravity in 3D with different density diffusion terms</figcaption>
</figure>
```

[`DensityDiffusionAntuono`](@ref) adds a correction term to solve this problem, but this
term is very expensive and adds about 40--50% of computational cost.

### API

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "weakly_compressible_sph", "density_diffusion.jl")]
```

## [Corrections](@id corrections)

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("general", "corrections.jl")]
```

## [Surface Tension](@id surface_tension)

### Akinci-based intra-particle force surface tension and wall adhesion model
The work by Akinci proposes three forces:
- a cohesion force
- a surface area minimization force
- a wall adhesion force

The classical model is composed of the curvature minimization and cohesion force.

#### Cohesion force
The model calculates the cohesion force based on the distance between particles and the support radius ``h_c``.
This force is determined using two distinct regimes within the support radius:
- For particles closer than half the support radius,
  a repulsive force is calculated to prevent particle clustering too tightly,
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

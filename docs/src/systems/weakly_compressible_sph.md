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

## [Density Diffusion](@id density_diffusion)

Density diffusion can be used with [`ContinuityDensity`](@ref) to remove the noise in the
pressure field. It is highly recommended to use density diffusion when using WCSPH.

### Formulation

All density diffusion terms extend the continuity equation (see [`ContinuityDensity`](@ref))
by an additional term
```math
\frac{\mathrm{d}\rho_a}{\mathrm{d}t} = \sum_{b} m_b v_{ab} \cdot \nabla W_{ab}
    + \delta h c \sum_{b} V_b \psi_{ab} \cdot \nabla W_{ab},
```
where ``V_b = m_b / \rho_b`` is the volume of particle ``b`` and ``\psi_{ab}`` depends on
the density diffusion method (see
[`AbstractDensityDiffusion`](@ref TrixiParticles.AbstractDensityDiffusion) for available terms).
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

## [Particle Shifting Technique](@id shifting)

The Particle Shifting Technique (PST) is used to correct tensile instability
in regions of low pressure, as observed in viscous flow around an object.
Without PST, tensile instability causes non-physical separation of the fluid
from the object, introducing void regions behind the object.

At lower resolutions, PST alone can be effective to correct a viscous flow
around a cylinder, as shown in this figure.
![particle_shifting](https://github.com/user-attachments/assets/70892b0b-af36-4531-b328-f73e63a5e33c)
At higher resolutions, PST alone is not effective anymore; see the figure in
[Tensile Instability Control](@ref tic).
We recommend using PST and Tensile Instability Control together
in such simulations.

### Mathematical formulation

We use the following formulation by [Sun et al. (2018)](@cite Sun2018).
After each time step, a correction term ``\delta r_a`` is added to the position ``r_a``
of particle ``a``, which is given by
```math
\delta r_a = -4 \Delta t \, v_\text{max} h
    \sum_b \left( 1 + R \left( \frac{W_{ab}}{W(\Delta x_a)} \right)^n \right) \nabla W_{ab}
    \frac{m_b}{\rho_a + \rho_b},
```
where:
- ``\Delta t`` is the time step,
- ``v_\text{max}`` is the maximum velocity over all particles,
- ``h`` is the smoothing length,
- ``R`` and ``n`` are constants, which are set to ``0.2`` and ``4`` respectively,
- ``W(\Delta x_a)`` is the smoothing kernel of the particle size of particle ``a``,
  which can be interpreted as the target particle spacing that we want to achieve.
- ``\nabla W_{ab}`` is the gradient of the smoothing kernel,
- ``m_b`` is the mass of particle ``b``,
- ``\rho_a, \rho_b`` is the density of particles ``a`` and ``b``, respectively.

Note that we replaced ``\text{CFL} \cdot \text{Ma}`` by ``\Delta t \cdot v_\text{max} / h``,
as explained in [Sun2018](@cite Sun2018) on page 29, right above Equation 9.

The ``\delta``-SPH method (WCSPH with density diffusion) together with this formulation
of PST is commonly referred to as ``\delta^+``-SPH.

To apply particle shifting, use the keyword argument `shifting_technique` in the constructor
of a system that supports it.


## [Transport Velocity Formulation (TVF)](@id transport_velocity_formulation)

An alternative formulation is the so-called Transport Velocity Formulation (TVF)
by [Adami (2013)](@cite Adami2013).
[Ramachandran (2019)](@cite Ramachandran2019) applied the TVF also for the [EDAC](@ref edac)
scheme.

The transport velocity ``\tilde{v}_a`` of particle ``a`` is used to evolve the position
of the particle ``r_a`` from one time step to the next by
```math
\frac{\mathrm{d} r_a}{\mathrm{d}t} = \tilde{v}_a
```
and is obtained at every time step ``\Delta t`` from
```math
\tilde{v}_a (t + \Delta t) = v_a (t) + \Delta t \left(\frac{\tilde{\mathrm{d}} v_a}{\mathrm{d}t} - \frac{1}{\rho_a} \nabla p_{\text{background}} \right),
```
where ``\rho_a`` is the density of particle ``a`` and ``p_{\text{background}}``
is a constant background pressure field.
The tilde in the second term of the right-hand side indicates that the material derivative
has an advection part.

The discretized form of the last term is
```math
 -\frac{1}{\rho_a} \nabla p_{\text{background}} \approx  -\frac{p_{\text{background}}}{m_a} \sum_b \left(V_a^2 + V_b^2 \right) \nabla_a W_{ab},
```
where ``V_a``, ``V_b`` denote the volume of particles ``a`` and ``b`` respectively.
Note that although in the continuous case ``\nabla p_{\text{background}} = 0``,
the discretization is not 0th-order consistent for **non**-uniform particle distribution,
which means that there is a non-vanishing contribution only when particles are disordered.
That also means that ``p_{\text{background}}`` occurs as pre-factor to correct
the trajectory of a particle resulting in uniform pressure distributions.
Suggested is a background pressure which is in the order of the reference pressure,
but it can be chosen arbitrarily large when the time-step criterion is adjusted.

The inviscid momentum equation with an additional convection term for a particle
moving with ``\tilde{v}`` is
```math
\frac{\tilde{\mathrm{d}} \left( \rho v \right)}{\mathrm{d}t} = -\nabla p +  \nabla \cdot \bm{A},
```
where the tensor ``\bm{A} = \rho v\left(\tilde{v}-v\right)^T`` is a consequence
of the modified advection velocity and can be interpreted as the convection of momentum
with the relative velocity ``\tilde{v}-v``.

The discretized form of the momentum equation for a particle ``a`` reads as
```math
\frac{\tilde{\mathrm{d}} v_a}{\mathrm{d}t} = \frac{1}{m_a} \sum_b \left(V_a^2 + V_b^2 \right) \left[ -\tilde{p}_{ab} \nabla_a W_{ab} + \frac{1}{2} \left(\bm{A}_a + \bm{A}_b \right) \cdot \nabla_a W_{ab} \right].
```
Here, ``\tilde{p}_{ab}`` is the density-weighted pressure
```math
\tilde{p}_{ab} = \frac{\rho_b p_a + \rho_a p_b}{\rho_a + \rho_b},
```
with the density ``\rho_a``, ``\rho_b`` and the pressure ``p_a``, ``p_b`` of particles ``a``
and ``b``, respectively. ``\bm{A}_a`` and ``\bm{A}_b`` are the convection tensors
for particle ``a`` and ``b``, respectively, and are given, e.g., for particle ``a``,
as ``\bm{A}_a = \rho v_a\left(\tilde{v}_a-v_a\right)^T``.

To apply the TVF, use the keyword argument `shifting_technique` in the constructor
of a system that supports it.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "shifting_techniques.jl")]
```


## [Tensile Instability Control](@id tic)

Tensile Instability Control (TIC) is a pressure acceleration formulation to correct tensile
instability in regions of low pressure, as observed in viscous flow around an object.
The technique was introduced by [Sun et al. (2018)](@cite Sun2018).
The formulation is described in Section 2.1 of this paper.
It can be used in combination with the [Particle Shifting Technique (PST)](@ref shifting)
to effectively prevent non-physical separation of the fluid from the object.

As can be seen in the following figure, TIC alone can cause instabilities
and does not improve the simulation.
PST alone can mostly prevent separation at lower resolutions.
A small void region is still visible, but quickly disappears.
The combination of PST and TIC prevents separation effectively.
![low_res](https://github.com/user-attachments/assets/5b30d440-8ca5-4c13-94d0-d110de2eb7cc)

At higher resolutions, PST alone is not effective to prevent separation, as can be seen
in the next figure.
Only the combination of PST and TIC is able to produce physical results.
![high_res](https://github.com/user-attachments/assets/674aec76-33e6-4ee3-bcd7-ba8c381a2775)

### Mathematical formulation

The force that particle ``a`` experiences from particle ``b`` due to pressure is given by
```math
f_{ab} = -m_a m_b \frac{p_a + p_b}{\rho_a \rho_b} \nabla W_{ab}
```
for the WCSPH method with [`ContinuityDensity`](@ref).

The TIC formulation changes this force to
```math
f_{ab} = -m_a m_b \frac{|p_a| + p_b}{\rho_a \rho_b} \nabla W_{ab}.
```
Note that this formulation is asymmetric and sacrifices conservation of linear and angular
momentum.

```@docs
tensile_instability_control
```

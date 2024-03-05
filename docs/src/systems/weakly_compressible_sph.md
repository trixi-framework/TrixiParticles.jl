# [Weakly Compressible SPH](@id wcsph)

Weakly compressible SPH as introduced by Monaghan (1994). This formulation relies on a stiff
[equation of state](@ref equation_of_state) that generates large pressure changes
for small density variations.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "weakly_compressible_sph", "system.jl")]
```

### References
- Joseph J. Monaghan. "Simulating Free Surface Flows in SPH".
  In: Journal of Computational Physics 110 (1994), pages 399--406.
  [doi: 10.1006/jcph.1994.1034](https://doi.org/10.1006/jcph.1994.1034)

## [Equation of State](@id equation_of_state)

The equation of state is used to relate fluid density to pressure and thus allow
an explicit simulation of the [WCSPH system](@ref WeaklyCompressibleSPHSystem).
The equation in the following formulation was introduced by Cole (Cole 1948, pp. 39 and 43).
The pressure ``p`` is calculated as
```math
    p = B \left(\left(\frac{\rho}{\rho_0}\right)^\gamma - 1\right) + p_{\text{background}},
```
where ``\rho`` denotes the density, ``\rho_0`` the reference density,
and ``p_{\text{background}}`` the background pressure, which is set to zero when applied to
free-surface flows (Adami et al., 2012).

The bulk modulus, ``B =  \frac{\rho_0 c^2}{\gamma}``, is calculated from the artificial
speed of sound ``c`` and the isentropic exponent ``\gamma``.

An ideal gas equation of state with a linear relationship between pressure and density can
be obtained by choosing `exponent=1`, i.e.
```math
    p = B \left( \frac{\rho}{\rho_0} -1 \right) = c^2(\rho - \rho_0).
```

For higher Reynolds numbers, `exponent=7` is recommended, whereas at lower Reynolds
numbers `exponent=1` yields more accurate pressure estimates since pressure and
density are proportional.

When using [`SummationDensity`](@ref) (or [`DensityReinitializationCallback`](@ref))
and free surfaces, initializing particles with equal spacing will cause underestimated
density and therefore strong attractive forces between particles at the free surface.
Setting `clip_negative_pressure=true` can avoid this.
```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "weakly_compressible_sph", "state_equations.jl")]
```

## References
- Robert H. Cole. "Underwater Explosions". Princeton University Press, 1948.
- J. P. Morris, P. J. Fox, Y. Zhu
  "Modeling Low Reynolds Number Incompressible Flows Using SPH ".
  In: Journal of Computational Physics , Vol. 136, No. 1, pages 214--226.
  [doi: 10.1006/jcph.1997.5776](https://doi.org/10.1006/jcph.1997.5776)
- S. Adami, X. Y. Hu, N. A. Adams.
  "A generalized wall boundary condition for smoothed particle hydrodynamics".
  In: Journal of Computational Physics 231, 21 (2012), pages 7057–7075.
  [doi: 10.1016/J.JCP.2012.05.005](https://doi.org/10.1016/J.JCP.2012.05.005)

## [Viscosity](@id viscosity_wcsph)

TODO: Explain viscosity.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "viscosity.jl")]
```

## Density Diffusion

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
  <img src="https://lh3.googleusercontent.com/drive-viewer/AK7aPaBL-tqW6p9ry3NHvNnHVNufRfh_NSz0Le4vJ4n2rS-10Vr3Dkm2Cjb4T861vk6yhnvqMgS_PLXeZsNoVepIfYgpw-hlgQ=s1600" alt="density_diffusion_2d"/>
  <figcaption>Dam break in 2D with different density diffusion terms</figcaption>
</figure>
```

```@raw html
<figure>
  <img src="https://lh3.googleusercontent.com/drive-viewer/AK7aPaDKc0DCJfFH606zWFkjutMYzs70Y4Ot_33avjcIRxV3xNbrX1gqx6EpeSmysai338aRsOoqJ8B1idUs5U30SA_o12OQ=s1600" alt="density_diffusion_3d"/>
  <figcaption>Dam break in 3D with different density diffusion terms</figcaption>
</figure>
```

The simpler terms [`DensityDiffusionMolteniColagrossi`](@ref) and
[`DensityDiffusionFerrari`](@ref) do not solve the hydrostatic problem and lead to incorrect
solutions in long-running steady-state hydrostatic simulations with free surfaces
(Antuono et al., 2012). This can be seen when running the simple rectangular tank example
until ``t = 40`` (again using ``δ = 0.1``):

```@raw html
<figure>
  <img src="https://lh3.googleusercontent.com/drive-viewer/AK7aPaCf1gDlbxkQjxpyffPJ-ijx-DdVxlwUVb_DLYIW4X5E0hkDeJcuAqCae6y4eDydgTKe752zWa08tKVL5yhB-ad8Uh8J=s1600" alt="density_diffusion_tank"/>
  <figcaption>Tank in rest under gravity in 3D with different density diffusion terms</figcaption>
</figure>
```

[`DensityDiffusionAntuono`](@ref) adds a correction term to solve this problem, but this
term is very expensive and adds about 40--50% of computational cost.

### References
- M. Antuono, A. Colagrossi, S. Marrone.
  "Numerical Diffusive Terms in Weakly-Compressible SPH Schemes."
  In: Computer Physics Communications 183.12 (2012), pages 2570--2580.
  [doi: 10.1016/j.cpc.2012.07.006](https://doi.org/10.1016/j.cpc.2012.07.006)

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

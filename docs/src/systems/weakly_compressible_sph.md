# Weakly Compressible SPH

TODO: Explain the WCSPH formulation.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "weakly_compressible_sph", "system.jl")]
```

## State Equations

TODO: Explain how WCSPH uses the state equation.

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

## Corrections

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("general", "corrections.jl")]
```

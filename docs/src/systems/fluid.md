# [Fluid Models](@id fluid_models)

Currently available fluid methods are the [weakly compressible SPH method](@ref wcsph) and the
[entropically damped artificial compressibility for SPH](@ref edac).
This page lists models and techniques that apply to both of these methods.

## [Viscosity](@id viscosity_sph)

Viscosity is a critical physical property governing momentum diffusion within a fluid.
In the context of SPH, viscosity determines how rapidly velocity gradients are smoothed out,
influencing key flow characteristics such as boundary layer formation, vorticity diffusion,
and dissipation of kinetic energy. It also helps determine whether a flow is laminar or turbulent
under a given set of conditions.

Implementing viscosity correctly in SPH is essential for producing physically accurate results,
and different methods exist to capture both numerical stabilization and true viscous effects.

### Artificial (numerical) viscosity

Artificial (numerical) viscosity is a technique used to stabilize simulations of inviscid flows,
which would otherwise show unphysical particle movement due to numerical instability.
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
should be mainly used for inviscid flows (Euler), artificial stabilization
or shock-capturing, for which Monaghan [Monaghan1989](@cite) originally designed
this term to provide smoothing across shocks, intentionally overestimating the physical viscosity.
The implementation includes a dissipation term that becomes more significant
as particles approach one another. This helps suppress tensile instabilities,
which can lead to particle clumping and effectively smooths out high-frequency pressure fluctuations.
This increase in dissipation is triggered by the relative motion between particles:
as particles come closer and compress the local flow,
the artificial viscosity term becomes stronger to damp out rapid changes
and prevent unphysical clustering.
This ensures that while the simulation remains stable in challenging
flow regimes with large density or pressure variations,
the physical behavior is not overly altered.
Several extensions have been proposed to limit the dissipation effect for example
by Balsara ([Balsara1995](@cite)) or Morris ([Morris1997](@cite)).

##### Mathematical Formulation

The force exerted by particle ``b`` on particle ``a`` due to artificial viscosity is given by:

```math
F_{ab}^{\text{AV}} = - m_a m_b \Pi_{ab} \nabla W_{ab}
```

where:

- ``\Pi_{ab}`` is the artificial viscosity term defined as:
  ```math
  \Pi_{ab} =
  \begin{cases}
      -\frac{\alpha c \mu_{ab} + \beta \mu_{ab}^2}{\bar{\rho}_{ab}} & \text{if } \mathbf{v}_{ab} \cdot \mathbf{r}_{ab} < 0, \\
      0 & \text{otherwise}
  \end{cases}
  ```
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

To ensure that the simulation maintains a consistent Reynolds number when the resolution changes, the parameter ``\alpha`` must be adjusted accordingly.
Monaghan (2005) introduced an effective physical kinematic viscosity ``\nu`` defined as:

```math
\nu = \frac{\alpha h c}{2d + 4},
```

where **``d``** is the number of spatial dimensions. This relation allows the calibration of ``\alpha`` to achieve the desired viscous behavior as the resolution or simulation conditions vary.

#### ViscosityMorris

`ViscosityMorris` is ideal for moderate to low Mach number flows where accurately modeling physical viscous behavior is essential.
Developed by [Morris (1997)](@cite Morris1997) and later applied by [Fourtakas (2019)](@cite Fourtakas2019),
this method directly simulates the viscous stresses found in fluids rather than relying on artificial viscosity.
By approximating momentum diffusion based on local fluid properties, the method captures the actual viscous forces without excessive damping.
This results in a more realistic representation of flow dynamics in weakly compressible scenarios.

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

`ViscosityAdami`, introduced by [Adami (2012)](@cite Adami2012), is optimized for incompressible or weakly compressible flows where precise modeling of shear stress is critical.
It enhances boundary layer representation by better resolving shear gradients, increasing dissipation in regions with steep velocity differences (e.g., near solid boundaries)
while minimizing compressibility effects. This results in accurate laminar flow simulations and accurate physical shear stresses.

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

#### ViscosityCarreauYasuda

`ViscosityCarreauYasuda` implements the Carreau–Yasuda non-Newtonian viscosity model,
originally proposed by [Carreau (1972)](@cite Carreau1972) and extended by
[Yasuda et al. (1981)](@cite Yasuda1981). In this model, the kinematic viscosity
depends on the local shear rate. This makes it suitable for shear-thinning and
shear-thickening fluids, such as polymer solutions or blood-like fluids.
Instead of prescribing a single constant viscosity, the apparent viscosity
smoothly transitions between a low-shear plateau and a high-shear plateau.

In SPH, this can be incorporated by evaluating a shear-rate-dependent
viscosity locally and using it in the standard viscous discretization. A Newtonian
fluid is recovered as a special case when the parameters are chosen such that the
viscosity becomes independent of the shear rate. ([Zhang et al. (2017)](@cite Zhang2017);
[Vahabi & Sadeghy (2014)](@cite VahabiSadeghy2014)).


##### Mathematical Formulation

In the Carreau–Yasuda model, the kinematic viscosity ``\nu`` depends on the shear-rate magnitude ``\dot\gamma`` as
```math
\nu(\dot\gamma) = \nu_\infty + (\nu_0 - \nu_\infty)
\left[ 1 + (\lambda \dot\gamma)^a \right]^{\frac{n-1}{a}}.
```
where

- ``\nu_0``: zero-shear kinematic viscosity,
- ``\nu_\infty``: infinite-shear kinematic viscosity,
- ``\lambda``: time constant,
- ``a``: Yasuda parameter,
- ``n``: power-law index (``n < 1`` for shear-thinning, ``n > 1`` for shear-thickening),
- ``\dot\gamma``: shear-rate magnitude.

In this implementation the shear-rate magnitude is approximated per particle pair as
``\dot\gamma \approx \frac{\lVert \mathbf{v}_{ab} \rVert}{\lVert \mathbf{r}_{ab} \rVert + \epsilon}``,
with ``\mathbf{v}_{ab}`` the relative velocity, ``\mathbf{r}_{ab}`` the position difference,
and ``\epsilon`` a small regularization parameter.

All viscosities here are kinematic viscosities (m²/s); dynamic viscosity is obtained internally
via ``\eta = \rho \nu``. A Newtonian fluid is recovered for ``n = 1`` and
``\nu_0 = \nu_\infty``

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

Surface normals characterize the local orientation of an interface. In SPH, this geometric
information can be used for interface detection and reconstruction, curvature estimation,
interfacial boundary conditions, and interfacial force models. The computed normal field is also
available for analysis and VTK output.

#### Color field and gradient-based surface normals

The surface normal at a particle can be derived from a color field, a scalar marker used to
distinguish phases or materials. Its gradient is perpendicular to the color-field level sets and
therefore provides an interface-normal estimate; its orientation depends on the chosen color
convention. For a free surface whose exterior phase is not represented by particles, truncation of
the kernel support creates the corresponding discrete color-field gradient.

The simplest SPH approximation of an unnormalized color-field normal, ``n_a``, is

```math
n_a = \sum_b m_b \frac{c_b}{\rho_b} \nabla_a W_{ab},
```

where:

- ``c_b`` is the color field value for particle ``b``,
- ``m_b`` is the mass of particle ``b``,
- ``\rho_b`` is the density of particle ``b``,
- ``\nabla_a W_{ab}`` is the gradient of the smoothing kernel ``W_{ab}`` with respect to particle ``a``.

TrixiParticles evaluates this sum over every interacting physical fluid system. A neighboring
fluid therefore contributes its `color_value` even when it does not compute its own normals.
Particle-packing preprocessing systems are excluded. At a free surface, particles in the
unrepresented exterior phase are absent from the sum.

```@eval
using CairoMakie

let
    coordinate = range(-2.0, 2.0, length=401)
    interface_width = 0.3
    colorfield = @. 0.5 * (1.0 - tanh(coordinate / interface_width))
    colorfield_gradient = @. -0.5 / interface_width /
                             cosh(coordinate / interface_width)^2

    fig = Figure(size=(1000, 430), fontsize=18)
    color_axis = Axis(fig[1, 1],
                      xlabel="signed distance s/h", ylabel="color field c",
                      title="Diffuse color-field transition")
    gradient_axis = Axis(fig[1, 2],
                         xlabel="signed distance s/h", ylabel="dc/d(s/h)",
                         title="Color-field gradient")

    lines!(color_axis, coordinate, colorfield, color=:steelblue, linewidth=3)
    lines!(gradient_axis, coordinate, colorfield_gradient, color=:darkorange,
           linewidth=3)
    vlines!(color_axis, [0.0], color=:black, linestyle=:dash, linewidth=2)
    vlines!(gradient_axis, [0.0], color=:black, linestyle=:dash, linewidth=2)
    hlines!(gradient_axis, [0.0], color=(:black, 0.35), linewidth=1)
    xlims!(color_axis, extrema(coordinate))
    xlims!(gradient_axis, extrema(coordinate))

    CairoMakie.save("colorfield_profile.png", fig)
end
```

![A diffuse color field and its gradient across an interface](colorfield_profile.png)

The color field is approximately constant within either phase. Its gradient is localized in the
transition region and vanishes away from the interface. The sign of the gradient determines the
normal orientation; exchanging the two color values reverses that orientation.

##### Multiple color values

With more than two color values, every transition between unequal values contributes to the
color-field gradient. The direction of each gradient points toward the larger color value, while
its magnitude depends on the size of the color jump.

This is useful when several represented fluid phases or materials need to remain distinguishable in a
single scalar field. The value of ``c`` identifies the local region, while ``\nabla c`` locates and
orients each interface. This information can support interface reconstruction, phase-specific
boundary conditions, and post-processing. Assigning the same color value to two adjacent regions
deliberately makes their common boundary invisible to the color-field gradient.

```@eval
using CairoMakie

let
    coordinate = range(-3.0, 3.0, length=601)
    interface_width = 0.18
    smooth_step(position) = @. 0.5 *
                               (1.0 + tanh((coordinate - position) / interface_width))
    smooth_step_gradient(position) = @. 0.5 / interface_width /
                                        cosh((coordinate - position) / interface_width)^2

    color_a, color_b, color_c = 0.0, 2.0, 1.0
    colorfield = color_a .+
                 (color_b - color_a) .* smooth_step(-1.0) .+
                 (color_c - color_b) .* smooth_step(1.0)
    colorfield_gradient = (color_b - color_a) .* smooth_step_gradient(-1.0) .+
                          (color_c - color_b) .* smooth_step_gradient(1.0)

    fig = Figure(size=(1000, 430), fontsize=18)
    color_axis = Axis(fig[1, 1],
                      xlabel="signed distance s/h", ylabel="color field c",
                      title="Three color values")
    gradient_axis = Axis(fig[1, 2],
                         xlabel="signed distance s/h", ylabel="dc/d(s/h)",
                         title="Interface gradients")

    lines!(color_axis, coordinate, colorfield, color=:steelblue, linewidth=3)
    lines!(gradient_axis, coordinate, colorfield_gradient, color=:darkorange,
           linewidth=3)
    text!(color_axis, -2.0, 0.3, text="A: c = 0", align=(:center, :center))
    text!(color_axis, 0.0, 1.5, text="B: c = 2", align=(:center, :center))
    text!(color_axis, 2.0, 0.7, text="C: c = 1", align=(:center, :center))
    vlines!(color_axis, [-1.0, 1.0], color=:black, linestyle=:dash, linewidth=2)
    vlines!(gradient_axis, [-1.0, 1.0], color=:black, linestyle=:dash, linewidth=2)
    hlines!(gradient_axis, [0.0], color=(:black, 0.35), linewidth=1)
    xlims!(color_axis, extrema(coordinate))
    xlims!(gradient_axis, extrema(coordinate))

    CairoMakie.save("multiple_color_values.png", fig)

    vertical_coordinate = range(-1.0, 1.0, length=101)
    colorfield_2d = repeat(reshape(colorfield, :, 1), 1, length(vertical_coordinate))
    particle_coordinates = [(x, y) for x in -2.75:0.25:2.75
                             for y in -0.8:0.25:0.8]
    particle_x = first.(particle_coordinates)
    particle_y = last.(particle_coordinates)

    normal_y = collect(range(-0.7, 0.7, length=5))
    normal_origins_x = vcat(fill(-1.0, length(normal_y)),
                            fill(1.0, length(normal_y)))
    normal_origins_y = vcat(normal_y, normal_y)
    normal_directions_x = vcat(fill(1.0, length(normal_y)),
                               fill(-1.0, length(normal_y)))
    normal_directions_y = zeros(length(normal_directions_x))

    normal_fig = Figure(size=(1000, 430), fontsize=18)
    normal_axis = Axis(normal_fig[1, 1], aspect=DataAspect(),
                       xlabel="x/h", ylabel="y/h",
                       title="Unit normals at multiple color interfaces")
    heatmap = heatmap!(normal_axis, coordinate, vertical_coordinate, colorfield_2d,
                       colormap=:viridis, colorrange=(0.0, 2.0))
    scatter!(normal_axis, particle_x, particle_y, color=(:black, 0.3), markersize=5)
    text!(normal_axis, -2.0, 0.88, text="A: c = 0", color=:white,
          align=(:center, :top))
    text!(normal_axis, 0.0, 0.88, text="B: c = 2", color=:black,
          align=(:center, :top))
    text!(normal_axis, 2.0, 0.88, text="C: c = 1", color=:white,
          align=(:center, :top))
    vlines!(normal_axis, [-1.0, 1.0], color=:white, linewidth=3)
    arrows2d!(normal_axis, normal_origins_x, normal_origins_y,
              normal_directions_x, normal_directions_y,
              normalize=true, lengthscale=0.45, color=:black,
              shaftwidth=3, tipwidth=14, tiplength=10)
    xlims!(normal_axis, extrema(coordinate))
    ylims!(normal_axis, extrema(vertical_coordinate))
    Colorbar(normal_fig[1, 2], heatmap, label="color field c")

    CairoMakie.save("multiple_color_surface_normals.png", normal_fig)
end
```

![A color field and its gradient for three different color values](multiple_color_values.png)

Here the regions from left to right have ``c_A=0``, ``c_B=2``, and ``c_C=1``. At the
``A-B`` interface, the positive gradient points from ``A`` to ``B``. At the ``B-C`` interface,
the negative gradient points from ``C`` to ``B``. The first peak is twice as large because
``|c_B-c_A|=2`` instead of ``|c_C-c_B|=1``. Thus, numerical color differences affect an
unnormalized color-field normal. Normalization removes this difference from the direction but not
from formulations that retain the raw gradient magnitude, including the Akinci area term and the
momentum-conserving Morris surface delta. Color contrasts must therefore be selected consistently
when those models are used.

![Resulting surface normals for three color values](multiple_color_surface_normals.png)

The unit-normal directions are defined only in the transition regions. At the ``A-B`` interface
they point to the right, from ``A`` toward the larger value in ``B``. At the ``B-C`` interface they
point to the left, again toward ``B``. Inside each constant-color region the gradient vanishes, so
the unit normal is undefined and is represented numerically by a zero vector. If the opposite
orientation is required, the normal sign must be reversed according to the chosen phase convention.

#### Normalization of surface normals

The color-field gradient ``n_a`` is generally not a unit vector. Formulations that require only
the interface orientation use the unit normal

```math
\hat{n}_a = \frac{n_a}{\Vert n_a \Vert}.
```

Normalization separates the interface orientation from the magnitude of the discrete color-field
gradient. In TrixiParticles, standalone analysis/VTK output and the Akinci surface-area force use
the filtered, unnormalized gradient. The Morris formulations use unit normals for curvature or
surface-stress calculations; the momentum-conserving formulation separately retains the raw
gradient magnitude as its surface delta function.

```@eval
using CairoMakie

let
    coordinate = range(-1.35, 1.35, length=241)
    radius = 0.85
    interface_width = 0.1
    colorfield = [0.5 * (1.0 - tanh((hypot(x, y) - radius) / interface_width))
                  for x in coordinate, y in coordinate]

    particle_spacing = 0.17
    particle_coordinates = [(x, y) for x in (-radius):particle_spacing:radius
                             for y in (-radius):particle_spacing:radius
                             if hypot(x, y) <= radius]
    particle_x = first.(particle_coordinates)
    particle_y = last.(particle_coordinates)

    angles = range(0.0, 2pi, length=13)[1:(end - 1)]
    normal_x = -cos.(angles)
    normal_y = -sin.(angles)
    interface_x = radius .* cos.(angles)
    interface_y = radius .* sin.(angles)

    fig = Figure(size=(760, 650), fontsize=18)
    axis = Axis(fig[1, 1], aspect=DataAspect(),
                xlabel="x/h", ylabel="y/h",
                title="Interface orientation from the color-field gradient")
    heatmap = heatmap!(axis, coordinate, coordinate, colorfield,
                       colormap=:viridis, colorrange=(0.0, 1.0))
    contour!(axis, coordinate, coordinate, colorfield,
             levels=[0.1, 0.9], color=(:white, 0.8), linewidth=1.5)
    contour!(axis, coordinate, coordinate, colorfield,
             levels=[0.5], color=:white, linewidth=3)
    scatter!(axis, particle_x, particle_y, color=(:black, 0.35), markersize=5)
    arrows2d!(axis, interface_x, interface_y, normal_x, normal_y,
              normalize=true, lengthscale=0.3, color=:black,
              shaftwidth=3, tipwidth=14, tiplength=10)
    xlims!(axis, extrema(coordinate))
    ylims!(axis, extrema(coordinate))
    Colorbar(fig[1, 2], heatmap, label="color field c")

    CairoMakie.save("colorfield_surface_normals.png", fig)
end
```

![Color-field level sets and interface-normal directions](colorfield_surface_normals.png)

The particle phase has ``c \approx 1`` and the exterior has ``c \approx 0``. Consequently,
``\nabla c`` and the displayed unit normals point toward increasing ``c``. The arrows are
perpendicular to the color-field level sets; reversing the color convention reverses the arrows
without changing the interface geometry.

#### Handling noise and errors in normal calculation

Away from an interface, the exact color-field gradient vanishes, but particle disorder and
incomplete kernel support can produce small or poorly resolved normal estimates. The
[`ColorfieldSurfaceNormal`](@ref) thresholds mitigate this as follows:

1. Normals with insufficient particle support are discarded.
2. `interface_threshold` rejects gradients for which the dimensionless magnitude
   ``R\lVert n\rVert`` does not exceed the configured cutoff, where ``R`` is the kernel support
   radius. This applies to standalone, Akinci, and Morris normals.
3. `ideal_density_threshold` optionally suppresses particles whose neighbor count is close to
   ideal full support. This heuristic is intended only for a free surface with an unrepresented
   exterior phase. It must remain zero for fully represented multiphase interfaces, where valid
   interface particles can have full support.
4. Curvature calculations use a corrected formulation to reduce errors near interface fringes.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "surface_normal_sph.jl")]
```

---

## [Surface Tension](@id surface_tension)

Surface tension is a key phenomenon in fluid dynamics, influencing the behavior of droplets, bubbles, and fluid interfaces.
In SPH, surface tension is modeled as forces arising due to surface curvature and relative particle movement, ensuring realistic
simulation of capillary effects, droplet coalescence, and fragmentation.

The surface tension coefficient ``\sigma`` is a physical parameter that quantifies the energy required to increase the surface area
of a fluid by a unit amount. A higher value of ``\sigma`` indicates that the fluid resists changes to its surface area more strongly,
causing droplets or bubbles to assume shapes (often spherical) that minimize their surface. In practice, ``\sigma`` can be measured
experimentally through techniques such as the pendant drop method, the Wilhelmy plate method, or the du Noüy ring method,
each of which relates a measurable force or change in shape to the fluid’s surface tension. For pure substances,
tabulated reference values of ``\sigma`` at given temperatures are commonly used, while for mixtures or complex fluids,
direct experimental measurements or values can be estimated from empirical equation (see [Poling](@cite Poling2001) or [Lange](@cite Lange2005)).
In the following table some values are shown for reference. The values marked with a '~' are complex mixtures that are estimated by an empirical equation (see [Poling](@cite Poling2001)).

| **Fluid**    | **Surface Tension (``\sigma``) [N/m at 20°C]** |
|--------------|----------------------------------------------:|
| **Gasoline**    | ~0.022   [Poling](@cite Poling2001)             |
| **Ethanol**     | 0.022386 [Lange](@cite Lange2005)               |
| **Acetone**     | 0.02402  [Lange](@cite Lange2005)               |
| **Mineral Oil** | ~0.030   [Poling](@cite Poling2001)             |
| **Olive Oil**   | 0.03303  [Hui](@cite Hui1992), [MeloEspinosa](@cite MeloEspinosa2014) |
| **Glycerol**    | 0.06314  [Lange](@cite Lange2005)               |
| **Water**       | 0.07288  [Lange](@cite Lange2005)               |
| **Mercury**     | 0.486502 [Lange](@cite Lange2005)               |

### Model configuration

All surface tension coefficients must be finite and non-negative. A zero coefficient disables
the fluid-fluid surface force. Wall adhesion is controlled independently by the boundary's
`adhesion_coefficient`.

`CohesionForceAkinci` only evaluates the pairwise cohesion and optional wall-adhesion forces.
It does not require surface normals or `reference_particle_spacing`. The full
`SurfaceTensionAkinci` model and both Morris models require a surface-normal method. When one
of these models is selected without an explicit method, `ColorfieldSurfaceNormal()` is used.
The full Akinci model also requires `AkinciFreeSurfaceCorrection`, which supplies the
particle-neighborhood correction in Equations 4 and 5 of Akinci et al.

!!! note "Akinci kernels in two dimensions"
    Akinci et al. published the cohesion and adhesion kernels for three dimensions. In two
    dimensions, TrixiParticles.jl uses an integral-matching extension: each radial 2D kernel
    has the same full-space integral as its published 3D counterpart. This convention is not
    part of the original model, but gives both kernels dimensions of ``L^{-d}`` in ``d``
    dimensions. Their products with particle mass are therefore independent of resolution at
    a fixed smoothing-length-to-spacing ratio. Akinci surface tension is supported in two and
    three dimensions only. Integral matching removes the resolution dependence of the pair
    kernels, but does not turn their numerical coefficients into physical values in N/m.

    To preserve the pairwise cohesion and adhesion contributions from a previous 2D
    configuration that used the 3D normalizations, scale the coefficients at its
    compact-support radius ``h_c`` as

    ```math
    \sigma_{\mathrm{new}} = \frac{627}{790h_c}\sigma_{\mathrm{old}}, \qquad
    \beta_{\mathrm{new}} = \frac{42}{65h_c}\beta_{\mathrm{old}}.
    ```

    The migrated coefficients can then be held fixed when changing the resolution. Since
    `SurfaceTensionAkinci` uses ``\sigma`` for both cohesion and the unchanged curvature term,
    this migration also changes their relative weight; full-model configurations may require
     additional calibration.

### [Akinci-based intra-particle force surface tension and wall adhesion model](@id akinci_ipf)

The [Akinci](@cite Akinci2013) model divides surface tension into distinct force components:

#### Cohesion force

The cohesion force captures the attraction between particles at the fluid interface, creating the effect of surface tension.
It is defined by the distance between particles and the support radius ``h_c``, using a kernel-based formulation.

**Key features:**

- Particles within half the support radius experience a repulsive force to prevent clustering.
- Particles beyond half the radius but within the support radius experience an attractive force to simulate cohesion.

Mathematically:

```math
F_{\text{cohesion}} = -\sigma m_b C_d(r) \frac{r}{\Vert r \Vert},
```

where the dimension-dependent cohesion kernel is

```math
C_d(r)=\frac{K_d}{h_c^{d+6}}
\begin{cases}
(h_c-r)^3 r^3, & \text{if } 2r > h_c, \\
2(h_c-r)^3 r^3 - \frac{h_c^6}{64}, & \text{if } r > 0 \text{ and } 2r \leq h_c, \\
0, & \text{otherwise.}
\end{cases}
\qquad
K_2=\frac{25280}{627\pi}, \quad K_3=\frac{32}{\pi}.
```

The 3D constant is the published normalization. The 2D constant is chosen such that

```math
\int_{\mathbb{R}^2} C_2(\Vert\bm{r}\Vert)\,\mathrm{d}A
= \int_{\mathbb{R}^3} C_3(\Vert\bm{r}\Vert)\,\mathrm{d}V
= \frac{79}{336}.
```

#### Surface area minimization force

The surface area minimization term models curvature reduction by using the difference between the
raw color gradients. In the implementation it is evaluated in acceleration form as

```math
a_{a,\text{area}} = -\sigma h_a (n_a - n_b),
```

where ``n_a`` and ``n_b`` are the unnormalized color gradients of the interacting particles and
``h_a`` is the smoothing length of particle ``a``. The factor ``h_a`` makes the color-normal term
dimensionless, consistent with the Akinci formulation.

#### Combined-force correction

To compensate for particle-neighborhood deficiency at a free surface, the cohesion and
curvature contributions are multiplied by the symmetric factor

```math
K_{ab} = \frac{2\rho_0}{\rho_a + \rho_b}.
```

[`AkinciFreeSurfaceCorrection`](@ref) implements this factor for the combined fluid-fluid
surface tension force. Section 4 of [Akinci et al. (2013)](@cite Akinci2013) also applies the
factor to viscosity for the same particle-deficiency reason. It does not modify pressure or wall
adhesion forces.

The published correction assumes that the density estimate reflects missing neighbors. With
[`SummationDensity`](@ref), ``\rho_a`` and ``\rho_b`` in ``K_{ab}`` are the current densities.
For [`ContinuityDensity`](@ref) in a [`WeaklyCompressibleSPHSystem`](@ref) or
[`EntropicallyDampedSPHSystem`](@ref), TrixiParticles.jl
reconstructs the auxiliary densities

```math
\widetilde{\rho}_a = \sum_b m_b W_{ab}
```

and uses ``\widetilde{\rho}_a`` and ``\widetilde{\rho}_b`` in ``K_{ab}``. For
[`SurfaceTensionAkinci`](@ref), the same reconstructed densities are used in the particle-volume
weights ``m_b / \rho_b`` of the color-field normals, matching the density-summation formulation
of the published model. Pressure and all other density-dependent terms continue to use the
integrated continuity density. The auxiliary sum includes dummy boundary particles, so a wall
that completes the particle neighborhood is not misclassified as a free surface. This extension
makes the correction independent of the selected density calculator at the cost of one additional
density-summation neighbor loop per update stage.

#### Wall adhesion force

This force models the interaction between fluid and solid boundaries, simulating adhesion effects at walls.
It uses a custom kernel with a peak at 0.75 times the support radius:

```math
F_{\text{adhesion}} = -\beta m_b A_d(r) \frac{r}{\Vert r \Vert},
```

where the dimension-dependent adhesion kernel is

```math
A_d(r) = \frac{b_d}{h_c^{d+1/4}}
\begin{cases}
\sqrt[4]{-\frac{4r^2}{h_c} + 6r - 2h_c}, & \text{if } 2r > h_c \text{ and } r \leq h_c, \\
0, & \text{otherwise.}
\end{cases}
\qquad
b_2=\frac{13}{1200}, \quad b_3=0.007.
```

Again, ``b_3`` is the published value and ``b_2`` matches the full-space integrals. In terms
of the dimensionless radial moments

```math
J_d = \int_{1/2}^{1} q^{d-1}\left[2(1-q)(2q-1)\right]^{1/4}\,\mathrm{d}q,
```

the beta-function identities

```math
J_2 = \frac{3}{8}B\!\left(\frac{5}{4},\frac{5}{4}\right), \qquad
J_3 = \frac{65}{224}B\!\left(\frac{5}{4},\frac{5}{4}\right)
```

give ``J_3/J_2=65/84`` and thus ``b_2=2b_3J_3/J_2=13/1200``.

---

### [Morris surface tension model](@id morris_csf)

The method described by [Morris](@cite Morris2000) estimates curvature by combining particle color gradients (see [`surface_normal`](@ref)) and smoothing functions to derive surface normals.
The computed curvature is then used to determine forces acting perpendicular to the interface.
While this method provides accurate surface tension forces, it does not explicitly conserve momentum.

In the Morris model, surface tension is computed based on local interface curvature ``\kappa`` and the unit surface normal ``\hat{n}.``
By estimating ``\hat{n}`` and ``\kappa`` at each particle near the interface, the surface tension force for particle a can be written as:

```math
F_{\text{surface tension}} = - \sigma \frac{\kappa_a}{\rho_a}\hat{n}_a
```

This formulation focuses directly on geometric properties of the interface, making it relatively straightforward to implement when a reliable interface detection
(e.g., a color function) is available. However, accurately estimating ``\kappa`` and ``n`` may require fine resolutions.

---

### [Morris-based momentum-conserving surface tension model](@id moriss_css)

In addition to the simpler curvature-based formulation, [Morris](@cite Morris2000) introduced a momentum-conserving approach.
This method treats surface tension forces as arising from the divergence of a stress tensor, ensuring exact conservation
of linear momentum and offering more robust behavior for high-resolution or long-duration simulations
where accumulated numerical error can be significant.

#### Stress tensor formulation

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

This divergence can be computed numerically in the SPH framework as

```math
\sum_b \frac{m_b}{\rho_a \rho_b} (S_a + S_b) \nabla W_{ab}
```

#### Advantages and limitations

While momentum conservation makes this model attractive, it requires additional computational effort and stabilization
techniques to address instabilities in high-density regions.

### API

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "surface_tension.jl")]
```

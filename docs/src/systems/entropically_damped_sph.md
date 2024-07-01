# [Entropically Damped Artificial Compressibility (EDAC) for SPH](@id edac)

As opposed to the [weakly compressible SPH scheme](weakly_compressible_sph.md), which uses an equation of state,
this scheme uses a pressure evolution equation to calculate the pressure
```math
\frac{\mathrm{d} p_a}{\mathrm{d}t} =  - \rho c_s^2 \nabla \cdot v + \nu \nabla^2 p,
```
which is derived by Clausen (2013). This equation is similar to the continuity equation (first term, see
[`ContinuityDensity`](@ref)), but also contains a pressure damping term (second term, similar to density diffusion
see [`DensityDiffusion`](@ref)), which reduces acoustic pressure waves through an entropy-generation mechanism.

The pressure evolution is discretized with the SPH method by Ramachandran (2019) as following:

The first term is equivalent to the classical artificial compressible methods, which are commonly
motivated by assuming the artificial equation of state ([`StateEquationCole`](@ref) with `exponent=1`)
and is discretized as
```math
- \rho c_s^2 \nabla \cdot v = \sum_{b} m_b \frac{\rho_a}{\rho_b} c_s^2 v_{ab} \cdot \nabla_{r_a} W(\Vert r_a - r_b \Vert, h),
```
where ``\rho_a``, ``\rho_b``,  ``r_a``, ``r_b``, denote the density and coordinates of particles ``a`` and ``b`` respectively, ``c_s``
is the speed of sound and ``v_{ab} = v_a - v_b`` is the difference in the velocity.

The second term smooths the pressure through the introduction of entropy and is discretized as
```math
\nu \nabla^2 p = \frac{V_a^2 + V_b^2}{m_a} \tilde{\eta}_{ab} \frac{p_{ab}}{\Vert r_{ab}^2 \Vert + \eta h_{ab}^2} \nabla_{r_a}
W(\Vert r_a - r_b \Vert, h) \cdot r_{ab},
```
where ``V_a``, ``V_b`` denote the volume of particles ``a`` and ``b`` respectively and ``p_{ab}= p_a -p_b``  is the difference in the pressure.

The viscosity parameter ``\eta_a`` for a particle ``a`` is given as
```math
\eta_a = \rho_a \frac{\alpha h c_s}{8},
```
where it is found in the numerical experiments of Ramachandran (2019) that ``\alpha = 0.5``
is a good choice for a wide range of Reynolds numbers (0.0125 to 10000).

!!! note
    > The EDAC formulation keeps the density constant and this eliminates the need for the continuity equation
    > or the use of a summation density to ﬁnd the pressure. However, in SPH discretizations, ``m/\rho``
    > is typically used as a proxy for the particle volume. The density of the ﬂuids can
    > therefore be computed using the summation density approach.
    Ramachandran (2019)


```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "entropically_damped_sph", "system.jl")]
```

### References
- Prabhu Ramachandran. "Entropically damped artiﬁcial compressibility for SPH".
  In: Computers and Fluids 179 (2019), pages 579--594.
  [doi: 10.1016/j.compfluid.2018.11.023](https://doi.org/10.1016/j.compfluid.2018.11.023)
- Jonathan R. Clausen. "Entropically damped form of artificial compressibility for explicit simulation of incompressible flow".
  In: American Physical Society 87 (2013), page 13309.
  [doi: 10.1103/PhysRevE.87.013309](http://dx.doi.org/10.1103/PhysRevE.87.013309)

## [Transport Velocity Formulation (TVF)](@id transport_velocity_formulation)
Standard SPH suffers from problems like tensile instability or the creation of void regions in the flow.
Adami et al (2013) modified the advection velocity and added an extra term in the momentum equation to avoid this problems.
The authors introduced the so called Transport Velocity Formulation (TVF) for WCSPH. Ramachandran (2019) et al applied the TVF
also for the [EDAC](@ref edac) scheme.

The transport velocity ``\tilde{v}_a`` of particle ``a`` is used to evolve the position of the particle ``r_a`` from one time step to the next by
```math
\frac{\mathrm{d} r_a}{\mathrm{d}t} = \tilde{v}_a
```
and is obtained at every time-step ``\Delta t`` from
```math
\tilde{v}_a (t + \Delta t) = v_a (t) + \Delta t \left(\frac{\tilde{\mathrm{d}} v_a}{\mathrm{d}t} - \frac{1}{\rho_a} \nabla p_{\text{background}} \right)
```

where ``\rho_a`` is the density of particle ``a`` and ``p_{\text{background}}`` is a constant background pressure field.

The discretized form of the last term is
```math
 -\frac{1}{\rho_a} \nabla p_{\text{background}} \approx  -\frac{p_{\text{background}}}{m_a} \sum_b \left(V_a^2 + V_b^2 \right) \nabla_a W_{ab}
```
Note that although ``\nabla p_{\text{background}} = 0``, the discretization is not 0th-order consistent for **non**-uniform particle distribution,
which means that there is a non-vanishing contribution only when particles are disordered.
That also means that ``p_{\text{background}}`` occurs as prefactor to correct the trajectory of a particle resulting in uniform pressure distributions.
Suggested is a background pressure which is on the order of the reference pressure but can be chosen arbitrarily large when time-step criterion is adjusted.

The inviscid momentum equation with an additional convection term for a particle moving with ``\tilde{v}`` is
```math
\frac{\tilde{\mathrm{d}} \left( \rho v \right)}{\mathrm{d}t} = -\nabla p +  \nabla \cdot \bm{A}
```
 where the tensor ``\bm{A} = \rho v\left(\tilde{v}-v\right)`` is a consequence of the modified
 advection velocity and can be interpreted as the convection of momentum with the relative velocity ``\tilde{v}-v``.

The discretized form of the momentum equation for a particle ``a`` reads as
```math
\frac{\tilde{\mathrm{d}} v_a}{\mathrm{d}t} = \frac{1}{m_a} \sum_b \left(V_a^2 + V_b^2 \right) \left[ -\tilde{p}_{ab} \nabla_a W_{ab} + \frac{1}{2} \left(\bm{A}_a + \bm{A}_b \right) \cdot \nabla_a W_{ab} \right]
```
where ``V_a``, ``V_b`` denote the volume of particles ``a`` and ``b`` respectively and ``v_{ab}= v_a -v_b``  is the relative velocity.
Here, ``\tilde{p}_{ab}`` is the density-weighted pressure
```math
\tilde{p}_{ab} = \frac{\rho_b p_a + \rho_a p_b}{\rho_a + \rho_b},
```
with the density  ``\rho_a``,  ``\rho_b`` and the pressure  ``p_a``,  ``p_b`` of particles ``a`` and ``b`` respectively.

The convection tensor is calculated as ``\bm{A} = \rho v (\tilde{v}-v)^T``.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "transport_velocity.jl")]
```

### References
- S. Adami, X. Y. Hu, N. A. Adams.
  "A transport-velocity formulation for smoothed particle hydrodynamics".
  In: Journal of Computational Physics 241, (2013), pages 292--307.
  [doi: 10.1016/j.jcp.2013.01.043](http://dx.doi.org/10.1016/j.jcp.2013.01.043)
- Prabhu Ramachandran. "Entropically damped artiﬁcial compressibility for SPH".
  In: Computers and Fluids 179 (2019), pages 579--594.
  [doi: 10.1016/j.compfluid.2018.11.023](https://doi.org/10.1016/j.compfluid.2018.11.023)

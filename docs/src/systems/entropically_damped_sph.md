# [Entropically Damped Artificial Compressibility (EDAC) for SPH](@id edac)

As opposed to the [weakly compressible SPH scheme](weakly_compressible_sph.md), which uses an equation of state,
this scheme uses a pressure evolution equation to calculate the pressure
```math
\frac{\mathrm{d} p_a}{\mathrm{d}t} =  - \rho c_s^2 \nabla \cdot v + \nu \nabla^2 p,
```
which is derived by [Clausen (2013)](@cite Clausen2013). This equation is similar to the continuity equation (first term, see
[`ContinuityDensity`](@ref)), but also contains a pressure damping term (second term, similar to density diffusion
see [`AbstractDensityDiffusion`](@ref TrixiParticles.AbstractDensityDiffusion)),
which reduces acoustic pressure waves through an entropy-generation mechanism.

The pressure evolution is discretized with the SPH method by [Ramachandran (2019)](@cite Ramachandran2019) as following:

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
where it is found in the numerical experiments of [Ramachandran (2019)](@cite Ramachandran2019) that ``\alpha = 0.5``
is a good choice for a wide range of Reynolds numbers (0.0125 to 10000).

!!! note
    > The EDAC formulation keeps the density constant and this eliminates the need for the continuity equation
    > or the use of a summation density to ﬁnd the pressure. However, in SPH discretizations, ``m/\rho``
    > is typically used as a proxy for the particle volume. The density of the ﬂuids can
    > therefore be computed using the summation density approach. [Ramachandran2019](@cite)


```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "entropically_damped_sph", "system.jl")]
```

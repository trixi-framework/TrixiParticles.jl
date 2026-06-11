# [Delta-ALE-SPH](@id delta_ale_sph)

The ``\delta``-ALE-SPH method by [Antuono et al. (2021)](@cite Antuono2021) combines
``\delta``-SPH density diffusion with an arbitrary Lagrangian-Eulerian particle shifting
formulation. Unlike the constant-mass ``\delta^+``-SPH approximation, the full model evolves
both particle density and particle mass.

For particle ``a``, the implemented system follows equation (13) of the paper:

```math
\begin{aligned}
\frac{\mathrm{d}\rho_a}{\mathrm{d}t}
  &= -\rho_a \langle \nabla\cdot(\mathbf{u}+\delta\mathbf{u})\rangle_a
     + \langle\nabla\cdot(\rho\delta\mathbf{u})\rangle_a + D^\rho_a,\\
\frac{\mathrm{d}m_a}{\mathrm{d}t}
  &= \frac{m_a}{\rho_a}\langle\nabla\cdot(\rho\delta\mathbf{u})\rangle_a
     + D^m_a,\\
\frac{\mathrm{d}(m_a\mathbf{u}_a)}{\mathrm{d}t}
  &= -\frac{m_a}{\rho_a}\langle\nabla p\rangle_a
     + \frac{m_a}{\rho_a}\langle\nabla\cdot\mathbf{T}_v\rangle_a
     + \frac{m_a}{\rho_a}
       \langle\nabla\cdot(\rho\mathbf{u}\otimes\delta\mathbf{u})\rangle_a
     + m_a\mathbf{g}.
\end{aligned}
```

The particle position is integrated with
``\mathrm{d}\mathbf{r}_a/\mathrm{d}t=\mathbf{u}_a+\delta\mathbf{u}_a``.
The shifting velocity is bounded by ``U_\mathrm{max}/2`` as specified in equations
(17)-(18). Density and mass diffusion use the shared coefficient `delta`, whose recommended
value in the paper is `0.1`.

```julia
fluid_system = DeltaALESPHSystem(initial_condition;
                                 smoothing_kernel=WendlandC2Kernel{2}(),
                                 smoothing_length=2particle_spacing,
                                 sound_speed=10.0,
                                 reference_density=1.0,
                                 maximum_velocity=1.0,
                                 delta=0.1)
```

The formulation integrates density and mass, so `SummationDensity` is not an option.
`maximum_velocity` is the expected maximum physical velocity used to define the shifting
Mach number in the paper.

!!! warning
    For free-surface simulations, the paper projects the shifting velocity onto the local
    free-surface tangent and disables diffusion for poorly sampled kernel supports.
    Free-surface detection is not yet available in TrixiParticles.jl, so these two
    free-surface modifications are not currently applied.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "delta_ale_sph", "system.jl")]
```

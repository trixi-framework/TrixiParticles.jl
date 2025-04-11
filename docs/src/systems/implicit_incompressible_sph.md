# [Implicit Incompressible SPH](@id iisph)

Implicit Incompressible SPH as introduced by [M. Ihmsen](@cite IHMSEN et al). This schemes relies on computing the pressure values, by iteratively solving a linear system with a relaxed jacobi system, to resolve the particles density deviation from the rest density. This method uses a linear system ``Ax=b`` where the pressure values that are used for the pressure acceleration are the unknown variable ``x``.  
It does not use a state equation to generate pressure values like the [weakly compressible SPH scheme](weakly_compressible_sph.md).


In order to get the formulation for the linear system we start by discretizing the Continuity equation ``\frac{D\rho}{Dt} = - \rho \nabla \cdot \mathbb{v}``.
For a particle ``i`` a forward difference method is used to discretizie the left hand side of the equation to get ``\frac{\rho_i(t+ \Delta t - \rho_i(t))}{\Delta t}``. The right hand side gets discretized by the difference formulation of the SPH discretitzation of the gradient ``\nabla \cdot \mathbb{v} = \frac{1}{\rho_i} \sum_j m_j \mathbb{v}_{ij} \nabla W_{ij}``, where ``\mathbb{v}_{ij}`` is equal to ``\mathbb{v}_i - \mathbb{v}_j``.
So all in all the following discretized version of the continuity equation for a particle ``i`` is achieved:
```\frac{\rho_i(t + \Delta t) - \rho_(t)}{\Delta t} = \sum_j m_j \mathbb{v}_{ij}(t+\Delta t) \nabla W_{ij} ```

For this implicit formular  the unknown velocities ``v_{ij}(t + \Delta t)`` are needed, which depend on the unknown pressure values ``p(t+\Delta t)``.
The unknown velocities are given by adding all pressure and non-pressure values is given by the following formula:
```v_i(t + \Delta t) = v_i(t) * \frac{\mathbb{F}_i^{adv}(t) + \mathbb{F}_i^p(t)}{m_i}```
``F_i^{adv}``are all non-pressure forces such as gravity, viscosiy, surface tension and more, while ``F_i^p(t)``are the pressure forces. 
The unknown pressure force is given by 
```\mathbb{F_i^p(t)} = m_i * \nabla p_i = m_i \sum_j m_j \left( \frac{p_i(t)}{\rho_i^2(t)} - \frac{p_j(t)}{\rho_j^2(t)} \right) \nabla W_{ij}```

Note that the IISPH is an incompressible fluid system, which means that the density of the fluid does not change over the time. By assuming that we have a fixed density value (the rest density ``\rho_0``) for all fluid particle over the whole time of the simulation, we want to have that the density value at the next time step ``\rho_i(t + \Delta t)`` is also this rest density. So we can plug in ``\rho_0``for ``\rho_i(t + \Delta t)`` in the formula above. 
The goal is to compute the pressure values to get the pressure acceleration that is needed to achieve the rest density for each particle in the next time step. At the moment these pressure values are unknown, but all the non-pressure forces are known in ``t``.
Therefore a predicted density gets calculated by an predicted velocity ``v_i^{adv}= v_i(t) + \Delta t \frac{\mathbb{F}_i^{adv}(t)}{m_i}``, which depends only on the non-pressure forces ``F_i^{adv}``. 

```\rho_i^{adv} = \rho_i(t) + \Delta t \sum_j m_j v_{ij}^{adv} \nabla W_{ij}(t)```

To achieve the rest density the unknown pressure forces need to resolve the compression through the non-pressure forces, that means that they have to resolve the deviation of the predicted density and the rest density. 
Therefore following equation needs to be fulfilled:
``` \Delta t ^2 \sum_j m_j \left(  \frac{\mathbb{F}_i^p(t)}{m_i} - \frac{\mathbb{F}_j^p(t)}{m_j} \right) \nabla W_{ij}(t) = \rho_0 - \rho_i^{adv}```












```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "implicit_incompressible_sph", "system.jl")]
```
# [Implicit Incompressible SPH](@id iisph)

Implicit Incompressible SPH (IISPH) as introduced by [Ihmsen et al. (2013)](@cite Ihmsen2013)
is a method that achieves incompressibility by solving the pressure Poisson equation.
The resulting linear system is iteratively solved with the relaxed Jacobi method.
Unlike the [weakly compressible SPH method](@ref wcsph), incompressible methods determine
pressure by enforcing the incompressibility constraint rather than using an equation of
state.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "implicit_incompressible_sph", "system.jl")]
```
## Derivation
To derive the linear system of the pressure Poisson equation, we start by discretizing the
continuity equation
```math
\frac{D\rho}{Dt} = - \rho \nabla \cdot \bm{v}.
```

For a particle ``i``, discretizing the left-hand side of the equation with a forward
difference yields

```math
\frac{\rho_i(t+ \Delta t) - \rho_i(t)}{\Delta t}.
```

The divergence in the right-hand side is discretized with the SPH discretization for
particle ``i`` as
```math
-\frac{1}{\rho_i} \sum_j m_j \bm{v}_{ij} \cdot \nabla W_{ij},
```
where ``\bm{v}_{ij} = \bm{v}_i - \bm{v}_j``.

Together, the following discretized version of the continuity equation for a particle ``i``
is achieved:
```math
\frac{\rho_i(t + \Delta t) - \rho_i(t)}{\Delta t}
= \sum_j m_j \bm{v}_{ij}(t+\Delta t) \cdot \nabla W_{ij}.
```

Note that the linear system is only solved for fluid particles, so ``i`` always represents
a fluid particle, and ``j`` represents all neighboring particles of ``i``.

The right-hand side contains the unknown velocities ``\bm{v}_{ij}(t + \Delta t)`` at the
next time step, making it an implicit formulation.

Using the semi-implicit Euler method, we can obtain the velocity in the next time step as
```math
\bm{v}_i(t + \Delta t) = \bm{v}_i(t) + \Delta t \frac{\bm{F}_i^\text{adv}(t) + \bm{F}_i^p(t)}{m_i},
```

where ``\bm{F}_i^{\text{adv}}`` denotes all non-pressure forces such as gravity, viscosity, surface
tension and more, while ``\bm{F}_i^p`` denotes the unknown pressure forces, which we
want to solve for.

Note that the IISPH is an incompressible method, which means that the density of the
fluid remains constant over time. By assuming a fixed reference density ``\rho_0`` for all
fluid particles over the whole simulation, the density value at the next time
step ``\rho_i(t + \Delta t)`` also has to be this rest density. So ``\rho_0`` can be plugged in
for ``\rho_i(t + \Delta t)`` in the equation above.

The goal is to compute the pressure values required to obtain the pressure acceleration that
ensures each particle reaches the rest density in the next time step. At the moment these
pressure values are unknown in ``t``, but all the non-pressure forces are already known.
Therefore a predicted velocity can be calculated, which depends only on the non-pressure
forces ``\bm{F}_i^{\text{adv}}``:

```math
\bm{v}_i^{\text{adv}}(t+\Delta t)= \bm{v}_i(t) + \Delta t \frac{\bm{F}_i^{\text{adv}}(t)}{m_i},
```
Using this predicted velocity and the continuity equation, a predicted density can be defined
in a similar way as

```math
\rho_i^{\text{adv}}(t + \Delta t)= \rho_i(t) + \Delta t \sum_j m_j \bm{v}_{ij}^{\text{adv}}(t+\Delta t) \cdot \nabla W_{ij}(t).
```

To achieve the rest density, the unknown pressure forces must counteract the compression
caused by the non-pressure forces. In other words, they must resolve the deviation between
the predicted density and the reference density.

Therefore, the following equation needs to be fulfilled:
```math
\Delta t ^2 \sum_j m_j \left(  \frac{\bm{F}_i^p(t)}{m_i} - \frac{\bm{F}_j^p(t)}{m_j} \right) \cdot \nabla W_{ij}(t) = \rho_0 - \rho_i^{\text{adv}}.
```

This expression is derived by substituting the reference density ``\rho_0`` for
``\rho_i(t+\Delta t)`` in the discretized continuity equation and inserting the definitions of
the predicted velocity ``\bm{v}_{ij}(t+\Delta t)`` and predicted density ``\rho_i^{\text{adv}}``.

The pressure force is defined as:

```math
\bm{F}_i^p(t) = -\frac{m_i}{\rho_i}  \nabla p_i = -m_i \sum_j m_j \left( \frac{p_i(t)}{\rho_i^2(t)} + \frac{p_j(t)}{\rho_j^2(t)} \right) \nabla W_{ij}.
```

Substituting this definition into the equation yields a linear system ``\bm{A}(t) \bm{p}(t) = \bm{b}(t)``
with one equation and one unknown pressure value for each particle.
For ``n`` particles, this is a linear system with ``n`` equations and ``n`` unknowns:

```math
\sum_j a_{ij} p_j = b_i = \rho_0 - \rho_i^{\text{adv}}.
```

To solve for the pressure values, a relaxed Jacobi scheme is used.
This is an iterative numerical method to solve a linear system ``Ap=b``. In each iteration, the
new values of ``p`` are obtained as computed as

```math
p_i^{(k+1)} = (1-\omega) p_i^{(k)} + \omega \left( \frac{1}{a_{ii}} \left( b_i - \sum_{j \neq i} a_{ij} p_j^{(k)} \right)\right).
```

Substituting the right-hand side, we obtain

```math
p_i^{l+1} = (1-\omega) p_i^l + \omega \frac{\rho_0 - \rho_i^{\text{adv}} - \sum_{j \neq i} a_{ij}p_j^l}{a_{ii}}.
```

Therefore the diagonal elements ``a_{ii}`` and the sum ``\sum_{j \neq i} a_{ij}p_j^l`` need to
be determined.
This can be done efficiently by separating the pressure force acceleration into two
components: One that describes the influence of particle ``i``'s own pressure on its
displacement, and one that describes the displacement due to the pressure values of the
neighboring particles ``j``.

The pressure acceleration is given by:
```math
\Delta t^2 \frac{\bm{F}_i^p}{m_i} = -\Delta t^2 \sum_j m_j \left( \frac{p_i}{\rho_i^2} + \frac{p_j}{\rho_j^2} \right)\nabla W_{ij} =  \underbrace{\left(- \Delta t^2 \sum_j \frac{m_j}{\rho_i^2} \nabla W_{ij} \right) }_{d_{ii}} p_i + \sum_j \underbrace{-\Delta t^2 \frac{m_j}{\rho_j^2} \nabla W_{ij}}_{d_{ij}}p_j.
```

The ``d_{ii}p_i`` value describes the displacement of particle ``i`` because of the particle ``i``
and ``d_{ij}p_j`` describes the influence from the neighboring particles ``j``.
Using these values, the linear system can be rewritten as

```math
\rho_0 - \rho_i^{\text{adv}} = \sum_j m_j \left( d_{ii}p_i + \sum_k d_{ik}p_k - d_{jj}p_j - \sum_k d_{jk}p_k \right) \cdot \nabla W_{ij},
```

where the first sum over ``k`` loops over all neighbor particles of ``i`` and
the second sum over ``k`` loops over the neighbor particles of ``j``
(which is a neighbor of ``i``).
So the sum over the neighboring pressure values ``p_j`` also includes the pressure values
``p_i``, since ``i`` is a neighbor of ``j``.
To separate this sum, it can be written as

```math
\sum_k d_{jk} p_k = \sum_{k \neq i} d_{jk} p_k + d_{ji} p_i.
```

With this separation, the equation for the linear system can again be rewritten as

```math
\rho_0 - \rho_i^{\text{adv}} = p_i \sum_j m_j ( d_{ii} - d_{ji}) \cdot \nabla W_{ij}  + \sum_j m_j \left ( \sum_k d_{ik} p_k - d_{jj} p_j - \sum_{k \neq i} d_{jk}p_k \right) \cdot \nabla W_{ij}.
```

In this formulation all coefficients that are getting multiplied with the pressure value ``p_i``
are separated from the other. The diagonal elements ``a_{ii}`` can therefore be defined as:

```math
a_{ii} = \sum_j m_j ( d_{ii} - d_{ji}) \cdot \nabla W_{ij}.
```

The remaining part of the equation represents the influence of the other pressure values ``p_j``.
Hence, the final relaxed Jacobi iteration takes the form:

```math
p_i^{l+1} = (1 - \omega) p_i^{l} + \omega \frac{1}{a_{ii}} \left( \rho_0 -\rho_i^{\text{adv}} - \sum_j m_j \left( \sum_k d_{ik} p_k^l - d_{jj} p_j^l - \sum_{k \neq i} d_{jk} p_k^l \right) \cdot \nabla W_{ij} \right).
```

Because interactions are local, limited to particles within the kernel support defined by
the smoothing length, each particle's pressure ``p_i`` depends only on its own value and
nearby particles.
Consequently, the matrix ``A`` is sparse (most entries are zero).

Note that this expression is not completely correct yet, since ``j`` represents all neighboring
particles of ``i`` and some values like ``d_{jj}`` are not defined for boundary particles.
The boundary handling is being explained in more detail later.

The diagonal elements ``a_{ii}`` get computed and stored at the beginning of the simulation step
and remain unchanged throughout the relaxed Jacobi iterations. The same applies for the
``d_{ii}``. The coefficients ``d_{ij}`` are computed, but not stored, as part of the calculation
of ``a_{ii}``.
During the Jacobi iterations, two loops over all particles are performed:
The first updates the values of ``\sum_j d_{ij}`` and the second computes the updated
pressure values ``p_i^{l+1}``.

The final pressure update follows the equation of the relaxed Jacobi scheme as stated above, with two
exceptions:
#### 1. Pressure clamping
To avoid negative pressure values, one can enable pressure clamping. In this case, the
pressure update is given by

```math
p_i^{l+1} = \max \left(0, (1-\omega) p_i^l + \omega \frac{\rho_0 - \rho_i^{\text{adv}} - \sum_{j \neq i} a_{ij}p_j^l}{a_{ii}}\right).
```

#### 2. Small diagonal elements
If the diagonal element ``a_{ii}`` becomes too small or even zero, the simulation may become
unstable. This can occur, for example, if a particle is isolated and receives little or no
influence from neighboring particles, or due to numerical cancellations. In such cases, the
updated pressure value is set to zero.

There are also other options, like setting ``a_{ii}`` to the threshold value if it is beneath
and then update with the usual equation, or just don't update the pressure value at all, and
continue with the old value. The numerical error introduced by this technique remains small,
as only isolated or almost isolated particles are affected.


## Boundary Handling

The previously introduced formulation did not distinguish between fluid and boundary
particles. To account for boundary interactions correctly, a few modifications to the previous
equations are required.

First, the discretized form of the continuity equation must be adapted for the case in which
a neighboring particle is a boundary particle. From now on, we distinguish between
neighboring fluid particles (indexed by ``f``) and neighboring boundary particles (indexed by
``b``). We continue to use the index ``j`` if we want to refer to all neighboring particles.

The updated discretized continuity equation becomes:

```math
\frac{\rho_i(t + \Delta t) - \rho_i(t)}{\Delta t} = \sum_f m_f \bm{v}_{if}(t+\Delta t) \cdot \nabla W_{if} + \sum_b m_b \bm{v}_{ib}(t+\Delta t) \cdot \nabla W_{ib}.
```

For a stationary wall, ``\bm{v}_{ib}`` reduces to the fluid velocity. For a wall with
prescribed motion, the predictor uses the physical wall velocity ``\bm{v}_b(t)`` at
the current evaluation time. The no-slip ghost velocity is used only for viscosity.
The predicted density ``\rho^{\text{adv}}`` is therefore:

```math
\rho_i^{\text{adv}} = \rho_i(t) + \Delta t \sum_f m_f \bm{v}_{if}^{\text{adv}} \cdot \nabla W_{if}(t) + \Delta t \sum_b m_b \left(\bm{v}_{i}^{\text{adv}} - \bm{v}_b(t)\right) \cdot \nabla W_{ib}(t).
```

This leads to the following updated formulation of the linear system:

```math
\Delta t^2 \sum_f m_f \left(  \frac{\bm{F}_i^p(t)}{m_i} - \frac{\bm{F}_f^p(t)}{m_f} \right) \cdot \nabla W_{if} + \Delta t^2 \sum_b m_b \frac{\bm{F}_i^p(t)}{m_i} \cdot \nabla W_{ib} = \rho_0 - \rho_i^{\text{adv}}.
```

Boundary particles do not accelerate in response to pressure, so ``F_b^p`` does not
appear in this equation. This also holds for prescribed wall motion.

The pressure force acting on a fluid particle is computed as:

```math
\bm{F}_i^p(t) = -m_i \sum_f m_f \left( \frac{p_i(t)}{\rho_i^2(t)} + \frac{p_f(t)}{\rho_f^2(t)} \right) \nabla W_{if}(t) - m_i \sum_b m_b \left( \frac{p_i(t)}{\rho_i^2(t)} + \frac{p_b(t)}{\rho_b^2(t)} \right) \nabla W_{ib}(t).
```

This also leads to an updated version of the equation for the diagonal elements:
```math
a_{ii} = \sum_f m_f ( d_{ii} - d_{fi}) \cdot \nabla W_{if} + \sum_b m_b d_{ii} \cdot \nabla W_{ib}.
```

From this point forward, the computation of the coefficients required for the Jacobi scheme
(such as ``d_{ii}``, ``d_{ij}`` etc.) depends on the specific boundary density evaluation method
used in the chosen boundary model.



### Pressure Mirroring
When using pressure mirroring, the pressure value ``p_b`` of a boundary particle in the equation
above is defined to be equal to the pressure of the corresponding fluid particle ``p_i``.
In other words, the boundary particle "mirrors" the pressure of the fluid particle interacting
with it. As a result, the coefficient that describes the influence of a particle's own
pressure value ``p_i`` must include a doubled contribution from each boundary particle.
Therefore, ``d_{ii}`` becomes

```math
d_{ii} = -\Delta t^2 \sum_f \frac{m_f}{\rho_i^2} \nabla W_{if}
         - 2\Delta t^2 \sum_b \frac{m_b}{\rho_i^2} \nabla W_{ib}.
```

The corresponding relaxed Jacobi iteration for pressure mirroring then becomes:

```math
\begin{align*}
p_i^{l+1} = (1 - \omega) p_i^l + \omega \frac{1}{a_{ii}} &\left( \rho_0 - \rho_i^{\text{adv}}
 - \sum_f m_f \left( \sum_k d_{ik} p_k^l - d_{ff}p_f^l - \sum_{k \neq i} d_{fk} p_k^l \right) \cdot \nabla W_{if} \right. \\
& \quad - \left. \sum_b m_b \left( \sum_f d_{if} p_f^l \right) \cdot \nabla W_{ib} \right).
\end{align*}
```

### Pressure Zeroing
If pressure zeroing is used instead, the pressure value of a boundary particle ``p_b``
is assumed to be zero. In the linear system, this removes the boundary pressure unknowns,
but boundary particles still contribute through the ``p_i/\rho_i^2`` part of the pressure
acceleration. Therefore ``d_{ii}`` is

```math
d_{ii} = -\Delta t^2 \sum_j \frac{m_j}{\rho_i^2} \nabla W_{ij}.
```

The corresponding relaxed Jacobi iteration reads

```math
\begin{align*}
p_i^{l+1} = (1 - \omega) p_i^l + \omega \frac{1}{a_{ii}} &\left( \rho_0 - \rho_i^{\text{adv}}
 - \sum_f m_f \left( \sum_k d_{ik} p_k^l - d_{ff}p_f^l - \sum_{k \neq i} d_{fk} p_k^l \right) \cdot \nabla W_{if} \right. \\
& \quad - \left. \sum_b m_b \left( \sum_f d_{if} p_f^l \right) \cdot \nabla W_{ib} \right).
\end{align*}
```

### [Pressure Extrapolation](@id iisph_pressure_extrapolation)
The density calculators [`AdamiPressureExtrapolation`](@ref) and [`BernoulliPressureExtrapolation`](@ref)
can also be used with IISPH. Neither method adds a wall pressure unknown to the PPE.
Instead, the wall pressure is a function of the fluid pressure,
```math
p_b = \sum_f E_{bf} p_f + e_b,
```
where ``E_{bf}`` are the normalized kernel weights of the [extrapolation](@ref boundary_models)
and ``e_b`` collects the hydrostatic and dynamic terms and the pressure offset.
For an incompressible fluid, this extrapolation is a boundary condition for the pressure.
Since the fluid density does not change, the wall density does not change either.
Walls that exert forces on an IISPH system must therefore not have a state equation,
and their density is their initial density.

Writing the density change caused by the wall pressure force
``-\sum_b m_b p_b / \rho_b^2 \nabla W_{ib}`` as ``\bm{M} \bm{p}_b``, the PPE becomes
```math
(\bm{A} + \bm{M} \bm{E}) \bm{p} = \bm{\rho}_0 - \bm{\rho}^{\text{adv}} - \bm{M} \bm{e}.
```
This system is solved with the relaxed Jacobi scheme by evaluating the extrapolation
in every iteration. Before computing ``\sum_j d_{ij} p_j^l``, the wall pressure ``p_b^l``
is extrapolated from the current pressure iterate ``p^l``. The wall particles then enter
the displacement of each fluid particle like the particles of [`PressureBoundaries`](@ref),
```math
\sum_j d_{ij} p_j^l = \sum_f d_{if} p_f^l + \sum_b d_{ib} p_b^l,
\qquad d_{ib} = -\Delta t^2 \frac{m_b}{\rho_b^2} \nabla W_{ib}.
```
This displacement enters the off-diagonal part of the rows of the particle, of its fluid
neighbors, and of [`PressureBoundaries`](@ref).
The diagonal elements ``a_{ii}`` are unchanged, so the dependence of ``p_b`` on ``p_i``
is lagged by one iteration. For one fluid neighbor with ``\rho_i = \rho_b``, the iteration
contracts with the factor ``|1 - 2\omega|``, which vanishes for the default ``\omega = 0.5``.

After the solve, the regular boundary update extrapolates the solved fluid pressure
for the force calculation, so the forces use the wall pressure of the solved PPE.
For example, with one fluid neighbor, zero gravity, and zero pressure offset,
the wall pressure equals the fluid pressure. The viscous force of the walls is part of
the prediction ``\rho^{\text{adv}}``. It uses the no-slip velocity of the walls, which
is computed from the current fluid velocity before the prediction.
A wall can also extrapolate the pressure of other fluid systems. Therefore, the solve runs
after the density corrections and pressure updates of all other systems.

With these walls, the solve differs from the standard IISPH solve in two ways:
- It starts from zero pressure instead of half the previous pressure, so that the solution
  does not depend on the previous RHS evaluation. In a resting water column with
  20 particles across and the default tolerance, the fluid came to rest within 5 seconds
  when starting from zero pressure, but not when starting from half the previous pressure.
- The termination criterion uses the average absolute residual, to which particles with
  zero pressure only contribute when they are overdense. Positive and negative errors do
  not cancel, so `max_error` approximately bounds the average density error of the actual
  RHS. The bound is not strict, since the residual is evaluated for the pressure before
  the last update of the iteration.

Both require more iterations than the standard solve.
To solve for the wall pressure as a PPE unknown, use [`PressureBoundaries`](@ref).

Without a free surface, the pressure level of the fluid is only fixed by the density
error, as these walls act like Neumann conditions on the pressure. A wall that compresses
a completely filled, closed domain therefore leads to a nearly singular pressure equation,
and the solver can reach `max_iterations` before the density tolerance.

The script `validation/general/iisph_wall_pressure.jl` checks this coupling.
With one fluid neighbor, the wall pressure equals the analytic solution of the pair.
In the first time steps of a water column, it reports the density error of the actual RHS,
the deviation of the wall pressure from the extrapolation of the solved fluid pressure,
iteration counts, and RHS timings. It also integrates a resting water column and a water
column in a tank that accelerates upward with ``g/2`` until the fluid rests relative to
the tank. At particle spacings 0.04 and 0.02, the time-averaged pressure of the first
wall layer deviates by at most 3.1 % from the hydrostatic value, and the load on the
bottom by at most 1.3 % from the weight of the fluid in the accelerated frame.

### Pressure Boundaries
The [`PressureBoundaries`](@ref) density calculator was introduced by
[Band et al. (2018)](@cite Band2018b) specifically for IISPH fluid systems and can therefore
only be used with IISPH.
In the standard IISPH method the PPE is solved only for fluid particles. The
pressure values for the boundary particles are then approximated, for example by using
pressure mirroring.
With `PressureBoundaries`, however, the linear system is extended to
include the boundary particles as well. This means that the pressure values of both the fluid
and the boundary particles are computed directly by solving the PPE.
The main advantage of this method is that boundary pressures emerge naturally from the
solution of the PPE, leading to a consistent pressure distribution across fluid and boundary
domains. This avoids artificial constraints such as forcing ``p_{b}=0`` or ``p_{b}=p_{f}``,
and results in more accurate handling of fluid-boundary interactions.

The key difference is that the pressure values of the boundary particles now have their own
PPE within the linear system. In other words, the boundary particle pressure ``p_b`` is
also solved as part of the linear system.
This leads to the following condition for the boundary particles ``b``:

```math
\Delta t^2 \sum_f m_f \left(  \frac{\bm{F}_i^p(t)}{m_i} - \frac{\bm{F}_f^p(t)}{m_f} \right) \cdot \nabla W_{if} + \Delta t^2 \sum_b m_b \left(  \frac{\bm{F}_i^p(t)}{m_i} - \frac{\bm{F}_b^p(t)}{m_b} \right) \cdot \nabla W_{ib} = \rho_0 - \rho_i^{\text{adv}}.
```

Note that in this case ``i`` is a boundary particle, ``f`` are its fluid neighbors, and ``b``
its boundary neighbors.

Since the pressure force for boundary particles is zero (as mentioned before), and because
in this case ``i`` and ``b`` are both boundary particles, the PPE simplifies to

```math
\Delta t^2 \sum_f m_f \left( -\frac{\bm{F}_f^p(t)}{m_f}\right) \cdot \nabla W_{if} = \rho_0 - \rho_i^{\text{adv}}.
```
If we substitute the definition of the pressure force from above, we obtain

```math
\Delta t^2 \sum_f m_f \left(  \sum_k m_k \left( \frac{p_f(t)}{\rho_f^2(t)} + \frac{p_k(t)}{\rho_k^2(t)} \right) \nabla W_{fk}\right) \cdot \nabla W_{if} = \rho_0 - \rho_i^{\text{adv}}.
```

where ``k`` represents all neighboring particles (fluid and boundary) of fluid particle ``f``.

Because the boundary particles do not directly contribute to their own pressure force (only
indirectly as neighbors of fluid particles), their ``d_{ii}`` values are zero. Therefore the
diagonal elements simplify to

```math
a_{ii} = - \sum_f m_f d_{fi} \cdot \nabla W_{if}.
```

The off-diagonal term ``\sum_{j \neq i} a_{ij} p_j`` in the relaxed Jacobi iteration for boundary
particles takes the following form

```math
\sum_{j \neq i} a_{ij} p_j = \sum_f m_f \left( - d_{ff} p_f - \sum_{k \neq i} d_{fk}p_k \right) \cdot \nabla W_{if}.
```

But not only the addition of the boundary particles to the linear system changes when using
pressure boundaries, also the PPE for the fluid particles is changing slightly.

Since the boundary particles have now their own pressure values, ``p_b`` is no longer
eliminated (as in pressure zeroing or pressure extrapolation with ``p_b=0``) nor simply
replaced by the fluid particle's pressure (as in pressure mirroring with ``p_b=p_i``).
Instead, ``p_b`` remains part of the PPE.

This has no effect on the diagonal elements ``a_{ii}``, which are identical to those of the
other density calculators. The ``d_{ii}`` values are also computed in the same way as for
pressure zeroing or pressure extrapolation.
However,  the off-diagonal term ``\sum_{j \neq i} a_{ij} p_j`` in the relaxed Jacobi iteration
changes slightly.
For pressure boundaries it takes the form
```math
\sum_{j \neq i} a_{ij} p_j = \sum_f m_f \left( \sum_k d_{ik} p_k - d_{ff} p_f - \sum_{k \neq i} d_{fk} p_k \right) \cdot \nabla W_{if}
+ \sum_b m_b \left( \sum_k d_{ik} p_k  \right) \cdot \nabla W_{ib},
```
where ``k`` represents all neighboring particles of ``i`` (both fluid and boundary).

```@docs
    PressureBoundaries
```

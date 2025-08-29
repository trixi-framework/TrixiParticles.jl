# [Implicit Incompressible SPH](@id iisph)

Implicit Incompressible SPH (IISPH) as introduced by [Ihmsen et al. (2013)](@cite ihmsen2013)
is a method that achieves incompressibility by solving the pressure Poisson equation.
The resulting linear system is iteratively solved with the relaxed Jacobi method.
IISPH does not use a state equation to compute pressure values like the [weakly compressible
SPH method](@ref wcsph).

To derive the linear system of the pressure Poisson equation, we start by discretizing the
continuity equation
```math
\frac{D\rho}{Dt} = - \rho \nabla \cdot \bm{v}.
```

For a particle ``i``, discretizing the left-hand side of the equation with a forward
difference yields

```math
\frac{\rho_i(t+ \Delta t - \rho_i(t))}{\Delta t}.
```

The divergence in the right-hand side is discretized with the SPH discretization for
particle ``i`` as
```math
-\frac{1}{\rho_i} \sum_j m_j \bm{v}_{ij} \nabla W_{ij},
```
where ``\bm{v}_{ij} = \bm{v}_i - \bm{v}_j``.

Together, the following discretized version of the continuity equation for a particle $i$
is achieved:
```math
\frac{\rho_i(t + \Delta t) - \rho_i(t)}{\Delta t} = \sum_j m_j \bm{v}_{ij}(t+\Delta t) \nabla W_{ij}.
```

The right-hand side contains the unknown velocities $\bm{v}_{ij}(t + \Delta t)$ at the
next time step, making it an implicit formulation.

Using the semi-implicit Euler method, we can obtain the velocity in the next time step as
```math
\bm{v}_i(t + \Delta t) = \bm{v}_i(t) + \Deltat t \frac{\bm{F}_i^\text{adv}(t) + \bm{F}_i^p(t)}{m_i}
```

$\bm{F}_i^{\text{adv}}$ denotes all non-pressure forces such as gravity, viscosity, surface
tension and more, while $\bm{F}_i^p(t)$ denotes the unknown pressure forces, which we
want to solve for.

Note that the IISPH is an incompressible fluid system, which means that the density of the
fluid remain constant over time. By assuming a fixed reference density $\rho_0$ for all
fluid particle over the whole time of the simulation, the density value at the next time
step $\rho_i(t + \Delta t)$ also has to be this rest density. So $\rho_0$ can be plugged in
for $\rho_i(t + \Delta t)$ in the formula above.

The goal is to compute the pressure values required to obtain the pressure acceleration that
ensures each particle reaches the rest density in the next time step. At the moment these
pressure values are unknown in $t$, but all the non-pressure forces are already known.
Therefore a predicted velocity can be calculated which depends only on the non-pressure
forces $\bm{F}_i^{\text{adv}}$:

```math
\bm{v}_i^{\text{adv}}= \bm{v}_i(t) + \Delta t \frac{\bm{F}_i^{\text{adv}}(t)}{m_i},
```
Using this predicted velocity and the continuity equation a predicted density can also be
determined.

```math
\rho_i^{\text{adv}} = \rho_i(t) + \Delta t \sum_j m_j \bm{v}_{ij}^{\text{adv}} \nabla W_{ij}(t)
```

To achieve the rest density, the unknown pressure forces must counteract the compression
caused by the non-pressure forces. In other words, they must resolve the deviation between
the predicted density and the reference density.

Therefore following equation needs to be fulfilled:
```math
\Delta t ^2 \sum_j m_j \left(  \frac{\bm{F}_i^p(t)}{m_i} - \frac{\bm{F}_j^p(t)}{m_j} \right) \nabla W_{ij}(t) = \rho_0 - \rho_i^{\text{adv}}
```

This expression is derived by substituting the reference density $\rho_0$ for
$\rho_i(t+\Delta t)$ in the discretized continuity equation and inserting the definitions of
the predicted velocity $\bm{v}_{ij}(t+\Delta t)$ and predicted density $\rho_i^{\text{adv}}$.

The pressure force is defined as:

```math
\bm{F}_i^p(t) = m_i * \nabla p_i = m_i \sum_j m_j \left( \frac{p_i(t)}{\rho_i^2(t)} - \frac{p_j(t)}{\rho_j^2(t)} \right) \nabla W_{ij}
```

Substituting this definition into the equation, leadsa linear system $\bm{A}(t) \bm{p}(t) = \bm{b}(t)$
with one equation and one unknown pressure value for each particle. So in total it is a
system with  n equations and n unknown variables for n particles.

```math
\sum_j a_{ij} p_i = b_i = \rho_0 - \rho_i^{\text{adv}}
```

This linear system must be solved in order to get the pressure values, which is done by
using a relaxed jacobi scheme.

This is a iterative numerical method to solve a linear system $Ax=b$. In each iteration the
new values of the variable $x$ get computed by the following formular

```math
x_i^{(k+1)} = (1-\omega) x_i^{(k)} + \omega \left( \frac{1}{a_{ii}} \left( b_i - \sum_{j \neq i} a_{ij} x_j^{(k)} \right)\right)
```

In the context of the linear system for the pressure values the formula is

```math
p_i^{l+1} = (1-\omega) p_i^l + \omega \frac{\rho_0 - \rho_i^{\text{adv}} \sum_{j \neq i} a_{ij}p_j^l}{a{ii}}
```

Therefore the diagonal elements $a_{ii}$ and the sum $\sum_{j \neq i} a_{ij}p_j^l$ need to
be determined.
This can be done efficiently by separating the pressure force acceleration into two
components: One that describes the influence of particle i's own pressure on its
displacement, and one that describes the displacement due to the pressure values of the
neighboring particles $j$.

The pressure acceleration is given by:
```math
\Delta t^2 \frac{\bm{F}_i^p}{m_i} = -\Delta t^2 \sum_j m_j \left( \frac{p_i}{\rho_i^2} + \frac{p_j}{\rho_j^2} \right)\nabla W_{ij} = \left( - \Delta t^2 \sum_j \frac{m_j}{\rho_i^2} \nabla W_{ij} \right) p_i + \sum_j - \Delta t^2 \frac{m_j}{\rho_j^2} \nabla W_{ij}p_j
```

The $d_{ii}p_i$ value describes the displacement of particle $i$ because of the particle $i$
and $d_{ij}p_j$ describes the influence from the neighboring particles $j$.
Using this new values the linear system can be rewritten as

```math
\rho_0 - \rho_i^{\text{adv}} = \sum_j m_j \left( d_{ii}p_i + \sum_j d_{ij}p_j - d_{jj}p_j - \sum_k d_{jk}p_k \right) \nabla W_{ij}
```

where $k$ stands for the neighbor particles of the neighbor particle $j$ from $i$.
So the sum over the neighboring pressure values $p_j$ also includes the pressure values $p_i$,
since $i$ is a neighbor of $j$
To separate this sum it can be written as

```math
\sum_k d_{jk} p_k = \sum_{k \neq i} d_{jk} p_k + d_{ji} p_i
```

With this separation the equation for the linear system can again be rewritten as

```math
\rho_0 - \rho_i^{\text{adv}} = p_i \sum_j m_j ( d_{ii} - d_{ij})\nabla W_{ij}  + \sum_j m_j \left ( \sum_j d_{ij} p_j - d_{jj} p_j - \sum_{k \neq i} d_{jk}p_k \right) \nabla W_{ij}
```

In this formulation all coefficients that are getting multiplied with the pressure value $p_i$
are separated from the other. The diagonal elements $a_{ii}$ can therefore be defined as:

```math
a_{ii} = \sum_j m_j ( d_{ii} - d_{ij})\nabla W_{ij}
```

The remaining part of the equation represents the influence of the other pressure values $p_j$
​
 . Hence, the final relaxed Jacobi iteration takes the form:

```math
p_i^{l+1} = (1 - \omega) p_i^{l} + \omega \frac{1}{a_{ii}} \left( \rho_0 -\rho_i^{\text{adv}} \sum_j m_j \left( \sum_j d_{ij} p_j^l - d_{jj} p_j^l - \sum_{k \neq i} d_{jk} p_k^l \right) \nabla W_{ij} \right).
```

Note that the pressure value of a particle $i$ depends only on its own current pressure
value, the pressure values of his neighboring particles, and the neighbor of those
neighboring particle. That means that the most entries of the matrix are zero (i.e the
system is sparse).

The diagonal elements $a_ii$ get computed and stored at the beginning of the simulation step
and remain unchanged throughout the relaxed Jacobi iterations. The same applies for the
$d_{ii}$. The coefficients $d_{ij}$ are computed, but not stored, as part of the calculation
of $a_{ii}$.
During the jacobi iterations, two loops over all particles are performed:
The first updates the values of $\sum_j d_{ij}$ and the second is to compute the updated
pressure values $p_i^{l+1}$.

The final pressure update follows the given formula of the relaxed jacobi scheme, with two
exceptions:
#### 1. Pressure clamping
To avoid negative pressure values, one can enable pressure clamping. In this case, the
pressure update is given by:

```math
\max(0, p_i^{l+1}) = (1-\omega) p_i^l + \omega \frac{\rho_0 - \rho_i{\text{adv}} \sum_{j \neq i} a_{ij}p_j^l}{a{ii}}
```

#### 2. Small diagonal elements
If the diagonal element $a_{ii}$ becomes too small or even zero, the simulation may become
unstable. This can occur, for example, if a particle is isolated and receives little or no
influence from neighboring particles, or due to numerical cancellations. In such cases, the
updated pressure value is set to zero.

There are also other options, like setting $a_{ii}$ to the threshold value is its beneath
and then updare with the known formula, or just don't update the pressure value at all, and
continue with the old value. By setting the pressure value to zero, the numerical error
through this can not be so big to mess up a whole simulation, as long as it doesn't happens
for too many particles.


## Boundary Handling

The previously introduced formulation only considered interactions between fluid particles
and neglected interactions between fluid and boundary particles. To account for boundary
interactions, a few modifications to the previous equations are required.

First, the discretized form of the continuity equation must be adapted for the case in which
a neighboring particle is a boundary particle. From now on, we distinguish between
neighboring fluid particles (indexed by $j$) and neighboring boundary particles (indexed by
$b$).

The updated discretized continuity equation becomes:

```math
\frac{\rho_i(t + \Delta t) - \rho_i(t)}{\Delta t} = \sum_j m_j \bm{v}_{ij}(t+\Delta t) \nabla W_{ij} + \sum_b m_b \bm{v}_{ib}(t+\Delta t) \nabla W_{ib}
```

Since boundary particles do not have an own velocity the difference between the fluid
particles velocity and the boundary particles velocity simplifies to just the fluid
particles velocity $\bm{v}_{ib}(t+\Delta t) = \bm{v}_{i}(t+\Delta t)$
Accordingly, the predicted density $\rho^{\text{adv}}$ also includes contributions from boundary
particles:

```math
\rho_i^{\text{adv}} = \rho_i (t) + \Delta t \sum_j m_j \bm{v}_{ij}^{\text{adv}} \nabla W_{ij}(t) + \Delta t \sum_b m_b \bm{v}_{i}^{\text{adv}} \nabla W_{ib}(t)
```

This leads to the following updated formulation of the linear system:

```math
\Delta t^2 \sum_j m_j \left(  \frac{\bm{F}_i^p(t)}{m_i} - \frac{\bm{F}_j^p(t)}{m_j} \right) \nabla W_{ij} + \Delta t^2 \sum_b m_b \frac{\bm{F}_i^p(t)}{m_i} \nabla W_{ib} = \rho_0 - \rho_i^{\text{adv}}
```

It is important to note that, because boundary particles have no velocity, the pressure
acceleration of a fluid particle does not directly depend on the pressure forces of boundary
particles. However, the pressure forces of both the particle itself and its neighboring
fluid particles depend on the pressure values of all neighboring particles—this includes
boundary particles. Therefore, the pressure of a fluid particle is indirectly influenced by
the pressure values of nearby boundary particles.

The pressure force acting on a fluid particle is computed as:

```math
\bm{F}_i^p(t) = -m_i \sum_j \left( \frac{p_i(t)}{\rho_i^2(t)} + \frac{p_j(t)}{\rho_j^2(t)} \right) \nabla W_{ij}(t) -m_i \sum_b \left( \frac{p_i(t)}{\rho_i^2(t)} + \frac{p_b(t)}{\rho_j^2(t)} \right) \nabla W_{ib}(t)
```

From this point forward, the computation of the coefficients required for the Jacobi scheme
(such as $d_{ii}$, $d_{ij}$ etc.) depends on the specific boundary density evaluation method
used in the chosen boundary model.



### Pressure Mirroring
When using pressure mirroring, the pressure value $p_b$ of a boundary particle is defined to
be equal to the pressure value of the corresponding fluid particle $p_i$.
 In other words, the boundary particle "mirrors" the pressure of the fluid particle
 interacting with it.
As a result, the coefficient that describes the influence of a particle's own pressure value
$p_i$
​must also include contributions from boundary particles. Therefore, the formula for
calculating the coefficient $d_{ii}$ must be adjusted as follows:

```math
d_{ii} = -\Delta t^2 \sum_j \frac{m_j}{\rho_i^2} \nabla W_{ij} - \Delta t^2 \sum_b \frac{m_b}{\rho_i^2} \nabla W_{ib}
```

The corresponding relaxed Jacobi iteration for pressure mirroring then becomes:

```math
p_i^{l+1} = (1 - \omega) p_i^l + \omega \frac{1}{a_{ii}} \left( \rho_0 - \rho_i^{\text{adv}} - \sum_j m_j \left( \sum_j d_{ij} p_j^l - d_{jj}p_j^l - \sum_{k \neq i} d_{jk} p_k^l \right) \nabla W_{ij} - \sum_b m_b \sum_j d_{ij} p_j^l \nabla W_{ij} \right)
```

### Pressure Zeroing
If pressure zeroing is used instead, the pressure value of a boundary particle $p_b$
​is assumed to be zero. Consequently, boundary particles do not contribute to the pressure
forces acting on fluid particles.
In this case, the computation of the coefficient $d_{ii}$ remains unchanged and is given by:

```math
d_{ii} = -\Delta t^2 \sum_j \frac{m_j}{\rho_i^2} \nabla W_{ij}
```

The formula for the relaxed Jacobi iteration remains the same as in the pressure mirroring
approach. However, the contribution from boundary particles vanishes due to their zero
pressure:

```math
p_i^{l+1} = (1 - \omega) p_i^l + \omega \frac{1}{a_{ii}} \left( \rho_0 - \rho_i^{\text{adv}} - \sum_j m_j \left( \sum_j d_{ij} p_j^l - d_{jj}p_j^l - \sum_{k \neq i} d_{jk} p_k^l \right) \nabla W_{ij} - \sum_b m_b \sum_j d_{ij} p_j^l \nabla W_{ij} \right)
```










@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "implicit_incompressible_sph", "system.jl")]
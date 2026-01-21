# [Time integration](@id time_integration)

TrixiParticles.jl uses a modular approach where time integration is just another module
that can be customized and exchanged.
The function [`semidiscretize`](@ref) returns an `ODEProblem`
(see [the OrdinaryDiffEq.jl docs](https://docs.sciml.ai/DiffEqDocs/stable/types/ode_types/)),
which can be integrated with [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl).

In particular, a [`DynamicalODEProblem`](https://docs.sciml.ai/DiffEqDocs/stable/types/dynamical_types/)
is returned, where the right-hand side is split into two functions, the `kick!`, which
computes the derivative of the particle velocities and the `drift!`, which computes
the derivative of the particle positions.
This approach allows us to use [specialized time integration methods that do not work with
general `ODEProblem`s](@ref symplectic_schemes).
Note that this is not a true `DynamicalODEProblem` where the kick does not depend
on the velocity. Therefore, not all integrators designed for `DynamicalODEProblem`s
will work (properly) (see [below](@ref kick_drift_kick)).
However, all integrators designed for general `ODEProblem`s can be used.

## Usage

After obtaining an `ODEProblem` from [`semidiscretize`](@ref), let us call it `ode`,
we can pass it to the function `solve` of OrdinaryDiffEq.jl.
For most schemes, we do the following:
```julia
using OrdinaryDiffEq
sol = solve(ode, Euler(),
            dt=1.0,
            save_everystep=false, callback=callbacks);
```
Here, `Euler()` should in practice be replaced by a more useful scheme.
`callbacks` should be a `CallbackSet` containing callbacks like the [`InfoCallback`](@ref).
For callbacks, please refer to [the docs](@ref Callbacks) and the example files.
In this case, we need to either set a reasonable, problem- and resolution-dependent
step size `dt`, or use the [`StepsizeCallback`](@ref), which overwrites the step size
dynamically during the simulation based on a CFL-number.
We always set `save_everystep=false`, or OrdinaryDiffEq.jl would return the solution vector
for every time step, writing massive amounts of data into the RAM for large simulations.
To visualize data for every time step, [callbacks](@ref Callbacks) can be used.

Some schemes, e.g. the two schemes `RDPK3SpFSAL35` and `RDPK3SpFSAL49` mentioned below,
support automatic time stepping, where the step size is determined automatically based on
error estimates during the simulation.
These schemes do not use the keyword argument `dt` and will ignore the step size set by
the [`StepsizeCallback`](@ref).
Instead, they will work with the tolerances `abstol` and `reltol`, which can be passed as
keyword arguments to `solve`. The default tolerances are `abstol=1e-6` and `reltol=1e-3`.

## Recommended schemes

A list of schemes for general `ODEProblem`s can be found
[here](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/).
We commonly use the following three schemes:
- `CarpenterKennedy2N54(williamson_condition=false)`: A five-stage, fourth order
  low-storage Runge-Kutta method designed by [Carpenter and Kennedy](@cite Carpenter1994)
  for hyperbolic problems.
- `RDPK3SpFSAL35()`: A 5-stage, third order low-storage Runge-Kutta scheme with embedded
  error estimator, optimized for compressible fluid mechanics [Ranocha2022](@cite).
- `RDPK3SpFSAL49()`: A 9-stage, fourth order low-storage Runge-Kutta scheme with embedded
  error estimator, optimized for compressible fluid mechanics [Ranocha2022](@cite).

## [Symplectic schemes](@id symplectic_schemes)

Symplectic schemes like the leapfrog method are often used for SPH.

### [Leapfrog kick-drift-kick](@id kick_drift_kick)

The kick-drift-kick scheme of the leapfrog method, updating positions ``u``
and velocities ``v`` with the functions ``\operatorname{kick}`` and ``\operatorname{drift}``,
reads:
```math
\begin{align*}
v^{1/2} &= v^0 + \frac{1}{2} \Delta t\, \operatorname{kick}(u^0, t^0), \\
u^1 &= u^0 + \Delta t\, \operatorname{drift} \left( v^{1/2}, t^0 + \frac{1}{2} \Delta t \right), \\
v^1 &= v^{1/2} + \frac{1}{2} \Delta t\, \operatorname{kick}(u^{1}, t^0 + \Delta t).
\end{align*}
```
In this form, it is also identical to the velocity Verlet scheme.
Note that this only works as long as ``\operatorname{kick}`` does not depend on ``v``, i.e.,
in the inviscid case.
Once we add viscosity, ``\operatorname{kick}`` depends on both ``u`` and ``v``.
Then, the calculation of ``v^1`` requires ``v^1`` and becomes implicit.

The way this scheme is implemented in OrdinaryDiffEq.jl as `VerletLeapfrog`,
the intermediate velocity ``v^{1/2}`` is passed to ``\operatorname{kick}`` in the last stage,
resulting in first-order convergence when the scheme is used in the viscid case.

!!! warning
    Please do not use `VelocityVerlet` and `VerletLeapfrog` with TrixiParticles.jl.
    They will require very small time steps due to first-order convergence in the viscid case.

### Leapfrog drift-kick-drift

The drift-kick-drift scheme of the leapfrog method reads:
```math
\begin{align*}
u^{1/2} &= u^0 + \frac{1}{2} \Delta t\, \operatorname{drift}(v^0, t^0), \\
v^1 &= v^0 + \Delta t\, \operatorname{kick} \left( u^{1/2}, t^0 + \frac{1}{2} \Delta t \right), \\
u^1 &= u^{1/2} + \frac{1}{2} \Delta t\, \operatorname{drift}(v^{1}, t^0 + \Delta t).
\end{align*}
```
In the viscid case where ``\operatorname{kick}`` depends on ``v``, we can add another
half step for ``v``, yielding
```math
\begin{align*}
u^{1/2} &= u^0 + \frac{1}{2} \Delta t\, \operatorname{drift}(v^0, u^0, t^0), \\
v^{1/2} &= v^0 + \frac{1}{2} \Delta t\, \operatorname{kick}(v^0, u^0, t^0), \\
v^1 &= v^0 + \Delta t\, \operatorname{kick} \left( v^{1/2}, u^{1/2}, t^0 + \frac{1}{2} \Delta t \right), \\
u^1 &= u^{1/2} + \frac{1}{2} \Delta t\, \operatorname{drift}(v^{1}, u^{1}, t^0 + \Delta t).
\end{align*}
```
This scheme is implemented in OrdinaryDiffEq.jl as `LeapfrogDriftKickDrift` and yields
the desired second order as long as ``\operatorname{drift}`` does not depend on ``u``,
which is always the case.

### Symplectic position Verlet

When the density is integrated (with [`ContinuityDensity`](@ref)), the density is appended
to ``v`` as an additional dimension, so all previously mentioned schemes treat the density
similar to the velocity.
The SPH code [DualSPHysics](https://github.com/DualSPHysics/DualSPHysics) implements
a variation of the drift-kick-drift scheme where the density is updated separately.
See [Domínguez et al. 2022, Section 2.5.2](@cite Dominguez2022).
In the following, we will call the derivative of the density ``R(v, u, t)``,
even though it is actually included in the ``\operatorname{kick}`` as an additional dimension.

This scheme reads
```math
\begin{align*}
u^{1/2} &= u^0 + \frac{1}{2} \Delta t\, \operatorname{drift}(v^0, u^0, t^0), \\
v^{1/2} &= v^0 + \frac{1}{2} \Delta t\, \operatorname{kick}(v^0, u^0, t^0), \\
\rho^{1/2} &= \rho^0 + \frac{1}{2} \Delta t\, R(v^0, u^0, t^0), \\
v^1 &= v^0 + \Delta t\, \operatorname{kick} \left( v^{1/2}, u^{1/2}, t^0 + \frac{1}{2} \Delta t \right), \\
\rho^1 &= \rho^0 \frac{2 - \varepsilon^{1/2}}{2 + \varepsilon^{1/2}}, \\
u^1 &= u^{1/2} + \frac{1}{2} \Delta t\, \operatorname{drift}(v^{1}, u^{1}, t^0 + \Delta t),
\end{align*}
```
where
```math
\varepsilon^{1/2} = - \frac{R(v^{1/2}, u^{1/2}, t^0 + \frac{1}{2} \Delta t)}{\rho^{1/2}} \Delta t.
```
This scheme is implemented in TrixiParticles.jl as [`SymplecticPositionVerlet`](@ref).

```@docs
    SymplecticPositionVerlet
```

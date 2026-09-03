# Callbacks

[`UpdateCallback`](@ref) is required for systems that keep mutable state between time
steps. In the current rigid-contact implementation, this applies when a
[`RigidContactModel`](@ref) uses tangential spring history. Rigid contact requires
`UpdateCallback(interval=1)` so history is advanced once after every accepted step.
`UpdateCallback(interval=N)` with `N > 1` and `UpdateCallback(dt=...)` are rejected for
these systems.

Contact history is updated from accepted endpoint states only. The initialization call uses
zero elapsed time, while later calls use `integrator.t - integrator.tprev` rather than the
next proposed `integrator.dt`. If history changes, the callback invalidates any cached FSAL
derivative so the next step evaluates contact forces from the new state. Exactly one
`UpdateCallback` must own this update; multiple update callbacks are rejected to prevent
advancing the same tangential displacement more than once per step.

```@autodocs
Modules = [TrixiParticles]
Pages = map(file -> joinpath("callbacks", file), readdir(joinpath("..", "src", "callbacks")))
```

# [Custom Quantities](@id custom_quantities)

The following pre-defined custom quantities can be used with the
[`SolutionSavingCallback`](@ref) and [`PostprocessCallback`](@ref).

```@autodocs
Modules = [TrixiParticles]
Pages = ["general/custom_quantities.jl"]
```

# Mechanical Work Calculator

The `MechanicalWorkCalculator` is a special custom quantity to be used with the
[`PostprocessCallback`](@ref).

```@autodocs
Modules = [TrixiParticles]
Pages = ["general/mechanical_work_calculator.jl"]
```

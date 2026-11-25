# Callbacks

```@autodocs
Modules = [TrixiParticles]
Pages = map(file -> joinpath("callbacks", file), readdir(joinpath("..", "src", "callbacks")))
```

# [Custom Quantities](@id custom_quantities)

The following pre-defined custom quantities can be used with the
[`SolutionSavingCallback`](@ref) and [`PostprocessCallback`](@ref).

```@autodocs
Modules = [TrixiParticles]
Pages = ["custom_quantities/custom_quantities.jl"]
```

# Mechanical Work and Thrust Calculators

The `MechanicalWorkCalculator` and `ThrustCalculator` are designed to be used as custom
quantities with a [`PostprocessCallback`](@ref).

```@autodocs
Modules = [TrixiParticles]
Pages = ["custom_quantities/mechanical_work_calculator.jl"]
```

# Structure Motion Calculator

Use `StructureMotionCalculator` to reconstruct the motion of a structure at a point in the
reference configuration. It can be passed as a custom quantity to
[`PostprocessCallback`](@ref) and [`SolutionSavingCallback`](@ref).

```@autodocs
Modules = [TrixiParticles]
Pages = ["custom_quantities/structure_motion_calculator.jl"]
```

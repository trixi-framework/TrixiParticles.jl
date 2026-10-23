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

The `StructureMotionCalculator` is a special custom quantity to be used with the
[`PostprocessCallback`](@ref) and [`SolutionSavingCallback`](@ref).

```@autodocs
Modules = [TrixiParticles]
Pages = ["custom_quantities/structure_motion_calculator.jl"]
```

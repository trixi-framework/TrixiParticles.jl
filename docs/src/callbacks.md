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
Pages = ["general/custom_quantities.jl"]
```

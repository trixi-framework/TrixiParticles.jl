# Callbacks

[`UpdateCallback`](@ref) is required for systems that keep mutable state between time
steps. In the current rigid-contact implementation, this applies when a
[`RigidContactModel`](@ref) uses tangential history, i.e. the rigid-wall friction path
needs the tangential displacement cache to be updated between steps.

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

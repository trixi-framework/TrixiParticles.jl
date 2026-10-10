# [Gravity](@id gravity)

Gravity models use the same numeric units as the simulation setup. By default
[`NewtonianGravity`](@ref) uses the unitless convention ``G = 1`` through
[`DEFAULT_GRAVITATIONAL_CONSTANT`](@ref). To work in another consistent unit
system, pass `gravitational_constant` explicitly. TrixiParticles does not depend
on Unitful.jl for gravity constants.

```@autodocs
Modules = [TrixiParticles]
Pages = [
    joinpath("general", "gravity.jl")
]
```

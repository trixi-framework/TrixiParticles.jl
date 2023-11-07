# Initial Condition

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("general", "initial_condition.jl")]
```

## Setups

```@autodocs
Modules = [TrixiParticles]
Pages = map(file -> joinpath("setups", file), readdir(joinpath("..", "src", "setups")))
```

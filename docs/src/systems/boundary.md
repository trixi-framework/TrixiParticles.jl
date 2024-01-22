# Boundary System

```@autodocs
Modules = [TrixiParticles]
Pages = map(file -> joinpath("boundary", file), readdir(joinpath("..", "src", "schemes", "boundary")))
```

## Dummy Particles

TODO: Explain what that is.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "boundary", "dummy_particles", "dummy_particles.jl")]
```

## Repulsive Particles

TODO: Explain what that is.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "boundary", "monaghan_kajtar", "monaghan_kajtar.jl")]
```

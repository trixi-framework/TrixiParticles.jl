# TrixiParticles.jl API

## Callbacks

```@autodocs
Modules = [TrixiParticles]
Pages = map(file -> joinpath("callbacks", file), readdir(joinpath("..", "src", "callbacks")))
```

## Containers

```@autodocs
Modules = [TrixiParticles]
Pages = map(file -> joinpath("containers", file), readdir(joinpath("..", "src", "containers")))
```

## Semidiscretization

```@autodocs
Modules = [TrixiParticles]
Pages = map(file -> joinpath("semidiscretization", file), readdir(joinpath("..", "src", "semidiscretization")))
```

## Setups

### File rectangular_tank.jl
```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("setups", "rectangular_tank.jl")]
```

### File rectangular_shape.jl
```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("setups", "rectangular_shape.jl")]
```

### File circular_shape.jl
```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("setups", "circular_shape.jl")]
```

## SPH

### File density_calculators.jl
```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("sph", "density_calculators.jl")]
```

### File neighborhood_search.jl
```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("sph", "neighborhood_search.jl")]
```

### File penalty_force.jl
```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("sph", "penalty_force.jl")]
```

### File smoothing_kernels.jl
```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("sph", "smoothing_kernels.jl")]
```

### File state_equations.jl
```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("sph", "state_equations.jl")]
```

### File viscosity.jl
```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("sph", "viscosity.jl")]
```

## Util

```@autodocs
Modules = [TrixiParticles]
Pages = ["util.jl"]
```

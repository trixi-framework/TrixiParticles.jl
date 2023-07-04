# TrixiParticles.jl API

## Callbacks

```@autodocs
Modules = [TrixiParticles]
Pages = map(file -> joinpath("callbacks", file), readdir(joinpath("..", "src", "callbacks")))
```

## General

### File density_calculators.jl
```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("general", "density_calculators.jl")]
```

### File initial_condition.jl
```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("general", "initial_condition.jl")]
```

### File neighborhood_search.jl
```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("general", "neighborhood_search.jl")]
```

### File semidiscretization.jl
```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("general", "semidiscretization.jl")]
```

### File smoothing_kernels.jl
```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("general", "smoothing_kernels.jl")]
```

## Schemes

### Boundary

#### File system.jl
```@autodocs
Modules = [TrixiParticles]
Pages = map(file -> joinpath("boundary", file), readdir(joinpath("..", "src", "schemes", "boundary")))
```

#### File dummy_particles.jl
```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "boundary", "dummy_particles", "dummy_particles.jl")]
```
#### File monaghan_kajtar.jl
```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "boundary", "monaghan_kajtar", "monaghan_kajtar.jl")]
```

### Weakly Compressible SPH

#### File system.jl
```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "weakly_compressible_sph", "system.jl")]
```

#### File state_equations.jl
```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "weakly_compressible_sph", "state_equations.jl")]
```

#### File viscosity.jl
```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "viscosity.jl")]
```

### Entropically damped artificial compressibility for SPH

#### File system.jl
```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "entropically_damped_sph", "system.jl")]
```

### Total Lagrangian SPH

#### File system.jl
```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "solid", "total_lagrangian_sph", "system.jl")]
```

#### File penalty_force.jl
```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "solid", "total_lagrangian_sph", "penalty_force.jl")]
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

## Util

```@autodocs
Modules = [TrixiParticles]
Pages = ["util.jl"]
```

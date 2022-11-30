# Pixie.jl API

## Callbacks

```@autodocs
Modules = [Pixie]
Pages = map(file -> joinpath("callbacks", file), readdir(joinpath("..", "src", "callbacks")))
```

## Containers

```@autodocs
Modules = [Pixie]
Pages = map(file -> joinpath("containers", file), readdir(joinpath("..", "src", "containers")))
```

## Semidiscretization

```@autodocs
Modules = [Pixie]
Pages = map(file -> joinpath("semidiscretization", file), readdir(joinpath("..", "src", "semidiscretization")))
```

## Setups

```@autodocs
Modules = [Pixie]
Pages = [joinpath("setups", "rectangular_tank.jl")]
```

## SPH

### File neighborhood_search.jl
```@autodocs
Modules = [Pixie]
Pages = [joinpath("sph", "neighborhood_search.jl")]
```

### File smoothing_kernels.jl
```@autodocs
Modules = [Pixie]
Pages = [joinpath("sph", "smoothing_kernels.jl")]
```

### Fluid Dynamics

#### File density_calculators.jl
```@autodocs
Modules = [Pixie]
Pages = [joinpath("sph", "fluid", "density_calculators.jl")]
```

#### File state_equations.jl
```@autodocs
Modules = [Pixie]
Pages = [joinpath("sph", "fluid", "state_equations.jl")]
```

#### File viscosity.jl
```@autodocs
Modules = [Pixie]
Pages = [joinpath("sph", "fluid", "viscosity.jl")]
```

## Util

```@autodocs
Modules = [Pixie]
Pages = ["util.jl"]
```

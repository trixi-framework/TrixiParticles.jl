# Pixie.jl API

## Callbacks

```@autodocs
Modules = [Pixie]
Pages = map(file -> joinpath("callbacks", file), readdir(joinpath("..", "src", "callbacks")))
```

## Setups

```@autodocs
Modules = [Pixie]
Pages = [joinpath("setups", "rectangular_tank.jl")]
```

## SPH

### File boundary_container.jl
```@autodocs
Modules = [Pixie]
Pages = [joinpath("containers", "boundary_container.jl")]
```

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

### File sph.jl
```@autodocs
Modules = [Pixie]
Pages = [joinpath("containers", "container.jl")]
```

### File state_equations.jl
```@autodocs
Modules = [Pixie]
Pages = [joinpath("sph", "fluid", "state_equations.jl")]
```

### File viscosity.jl
```@autodocs
Modules = [Pixie]
Pages = [joinpath("sph", "fluid", "viscosity.jl")]
```

## Util

```@autodocs
Modules = [Pixie]
Pages = ["util.jl"]
```

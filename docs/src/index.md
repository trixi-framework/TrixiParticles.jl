# Pixie.jl API

## Callbacks

```@autodocs
Modules = [Pixie]
Pages = map(file -> joinpath("callbacks", file), readdir(joinpath("..", "src", "callbacks")))
```

## SPH

### File boundary_conditions.jl
```@autodocs
Modules = [Pixie]
Pages = [joinpath("sph", "boundary_conditions.jl")]
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
Pages = [joinpath("sph", "sph.jl")]
```

### File state_equations.jl
```@autodocs
Modules = [Pixie]
Pages = [joinpath("sph", "state_equations.jl")]
```

### File viscosity.jl
```@autodocs
Modules = [Pixie]
Pages = [joinpath("sph", "viscosity.jl")]
```

## Util

```@autodocs
Modules = [Pixie]
Pages = ["util.jl"]
```

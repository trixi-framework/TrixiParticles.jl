```@autodocs
Modules = [TrixiParticles]
Pages = map(file -> joinpath("preprocessing", file), readdir(joinpath("..", "src", "preprocessing", "point_in_poly")))
```
```@autodocs
Modules = [TrixiParticles]
Pages = map(file -> joinpath("preprocessing", file), readdir(joinpath("..", "src", "preprocessing", "shapes")))
```

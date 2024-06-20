```@autodocs
Modules = [TrixiParticles]
Pages = map(file -> joinpath("preprocessing", file), readdir(joinpath("..", "src", "preprocessing", "point_in_poly")))
```
```@autodocs
Modules = [TrixiParticles]
Pages = map(file -> joinpath("preprocessing", file), readdir(joinpath("..", "src", "preprocessing", "shapes")))
```

### References
- Alec Jacobson, Ladislav Kavan, and Olga Sorkine-Hornung "Robust inside-outside segmentation using generalized winding numbers".
  In: ACM Transactions on Graphics, 32.4 (2013), pages 1--12.
  [doi: 10.1145/2461912.2461916](https://doi.org/10.1145/2461912.2461916)
- Kai Horman, Alexander Agathos "The point in polygon problem for arbitrary polygons".
  In: Computational Geometry, 20.3 (2001), pages 131--144.
  [doi: 10.1016/s0925-7721(01)00012-8](https://doi.org/10.1016/S0925-7721(01)00012-8)

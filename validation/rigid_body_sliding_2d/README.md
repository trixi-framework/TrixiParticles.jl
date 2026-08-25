# 2D Rigid-Body Sliding Validation

This validation reuses
`examples/structure/sliding_rigid_squares_friction_2d.jl` through `trixi_include` and compares
kinetic rigid-wall friction against the analytical stopping distance of a body sliding on a
horizontal plane,

```math
s = \frac{v_0^2}{2 \mu_k g}.
```

The comparison assumes that the body starts in contact, remains in the kinetic-slip regime
until stopping, and experiences a constant normal load equal to its weight. The script runs
the same rigid square with wall particle spacings `0.03` and `0.015`. It reports the stopping
distance error for each resolution, the final horizontal velocity, and the difference between
the two numerical stopping distances. Numerical trajectories are written to JSON and CSV in
`out/validation_result_rigid_body_sliding_2d_wall_spacing_*`; the analytical trajectory is
added to each JSON file for plotting.

The following files are provided:

1. `validation_rigid_body_sliding_2d.jl`: Runs the example at both wall resolutions and
   computes the stopping-distance and wall-resolution errors.
2. `plot_rigid_body_sliding_results.jl`: Optionally plots both numerical trajectories against
   the analytical trajectory after the validation has generated its output files.

Run the validation from the repository root with

```bash
JULIA_LOAD_PATH="@:$PWD:@stdlib" julia --project=test \
  validation/rigid_body_sliding_2d/validation_rigid_body_sliding_2d.jl
```

Then create the comparison plot with

```bash
JULIA_LOAD_PATH="@:$PWD:@stdlib" julia --project=test \
  validation/rigid_body_sliding_2d/plot_rigid_body_sliding_results.jl
```

# 2D Rigid-Body Sliding Validation

`validation_rigid_body_sliding_2d.jl` validates kinetic rigid-wall friction against the
analytical stopping distance of a body sliding on a horizontal plane,

```math
s = \frac{v_0^2}{2 \mu_k g}.
```

The comparison assumes that the body starts in contact, remains in the kinetic-slip regime
until stopping, and experiences a constant normal load equal to its weight. The script runs
the same rigid square with wall particle spacings `0.03` and `0.015`. It reports the stopping
distance error for each resolution, the final horizontal velocity, and the difference between
the two numerical stopping distances.

Run the validation from the repository root with

```bash
JULIA_LOAD_PATH="@:$PWD:@stdlib" julia --project=test \
  validation/rigid_body_sliding_2d/validation_rigid_body_sliding_2d.jl
```

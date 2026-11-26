# SPH correction validation

This PR covers periodic conditions: local fluid operators in 1D, 2D and 3D, and a
time-dependent 2D Taylor–Green vortex. Slip walls, no-slip walls and fluid–structure
interaction are the next three cases, in that order.

## Periodic operator tests

From the repository root, pass the dimension (default: 2):

```bash
julia --project=. validation/corrections/validation_corrections.jl 1
julia --project=. validation/corrections/validation_corrections.jl 2
julia --project=. validation/corrections/validation_corrections.jl 3
```

| Dimension | Particles per axis (`N`) | Output directory |
|:--|:--|:--|
| 1D | 12, 24, 48, 96 | `out/corrections_periodic/1d/` |
| 2D | 12, 24, 48, 96 | `out/corrections_periodic/` |
| 3D | 12, 24, 48 | `out/corrections_periodic/3d/` |

On a regular unit lattice, every particle contributes to the error, including those
whose neighbors lie across a periodic seam. The WCSPH system uses Wendland C6 at
`h = 2Δx`. The analytical field is `f = 2 + sin(2πx)`, adding `0.5 cos(2πy)` in 2D
and 3D and a further `0.25 sin(2πz)` in 3D. The tests compare:

- Raw and Shepard interpolation, and direct and difference gradients of `f`.
- Raw and Shepard summation density and continuity-density reinitialization against
  `ρ = 1000 [1 + 0.1 (f − 2)]`.
- Pressure acceleration against `−∇f/ρ`, and the residual acceleration for constant pressure.

The runner writes `correction_convergence.csv` and `.md`. Errors are relative L₂ norms,
except for constant pressure, where the target is zero and the error is absolute RMS
acceleration. From `N = 24` to `48`, the corrected difference-gradient orders are
1.986 (1D), 1.988 (2D) and 1.988 (3D). These describe local operators at fixed neighbor count.

Run `plot_correction_results.jl` with the same dimension in an environment with CSV,
DataFrames and Plots to create the PNG and offline HTML report. To customize a run:

```julia
using TrixiParticles
trixi_include(@__MODULE__, joinpath(validation_dir(), "corrections", "validation_corrections.jl");
              dimension=3, resolutions=(12, 24), output_directory="out/periodic_manual")
```

## Moving particles: 2D Taylor–Green vortex

The [vortex example](../../examples/fluid/taylor_green_vortex_2d.jl) provides exact velocity
and kinetic-energy decay on the periodic square. At each resolution, compare three gradient
treatments (uncorrected, `GradientCorrection`, and `MixedKernelGradientCorrection`), each
with and without Shepard density correction. The mixed treatment combines kernel and
gradient corrections in the gradient operator itself. All six cases use
the same initially regular lattice, EDAC summation density, standard Adami transport velocity,
`Re = 100` (`ν = 0.01`), Wendland C6 at `h = 2Δx`, and integration to `t = 0.5`.

Use the standard Adami viscosity regularization `ε = 0.01` at `16²` and `32²`, and
`ε = 0.001` at `64²`. The latter better matches the measured uncorrected velocity and
energy decay on the finer grid; `32²` favored the standard value in that baseline.
The regularization is fixed before comparing corrections on each grid.

In an environment with TrixiParticles, OrdinaryDiffEqLowStorageRK, CSV, DataFrames and Plots:

```bash
julia --threads=4 --project validation/corrections/validation_tgv_decay.jl
julia --project validation/corrections/plot_tgv_decay.jl
```

`out/corrections_periodic/tgv_2d/tgv_decay.csv` and `tgv_decay.png` show the mass-weighted
relative L₂ velocity error at the current particle positions and the measured energy ratio
against `K(t)/K(0) = exp(−16π²νt)`. Compare corrections within each resolution.

- `tgv_error_vs_resolution.png` compares the final and largest sampled velocity errors
  across `N = 16, 32, 64`. The `1/N` guide shows what first-order error reduction would
  look like: doubling `N` halves the error. Gradient and mixed corrections reproduce
  linear fields locally; that does not guarantee first-order decay of the full vortex
  error. The finest grid also has a different viscosity regularization, as noted above.
- `tgv_performance.png` shows median full-solve wall time (bars: min–max) and its ratio
  to the uncorrected solve on the same grid. Each case gets one warm-up and three timed
  solves from fresh initial conditions. Timing includes adaptive time stepping and saving
  the requested states in memory; it excludes setup, compilation during warm-up, error
  evaluation, and file I/O. `tgv_timings.csv` records individual timings, accepted steps,
  thread counts, CPU and Julia version. Use the same thread count on an idle machine.

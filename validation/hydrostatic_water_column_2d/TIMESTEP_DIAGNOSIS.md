# WCSPH summation-density timestep investigation

These sweeps predate [reference-consistent initialization](INITIAL_STATE.md).
Set `initialize_summation_density=false` when reproducing their earlier initial-state
formulation with the updated example.

## Finding

Smaller fluid steps substantially change the failure times, but the tested reductions
do not recover a stable hydrostatic trajectory. The finest runs still develop large
fluid motion and leave the particle grid. Independent initial-RHS measurements expose
a large free-surface force imbalance before time integration starts.

Dynamic tests use resolution `n=3`, with the boundary-density predictor applied.
The initial-RHS probe additionally checks `n=5`. These are targeted diagnostics of the
unclipped summation-density configuration, with no added artificial viscosity.

## Controlled timestep reductions

The geometry, material parameters, damping, Wendland C2 kernel, `h/Δx=√2`, Cole EOS
(`c=50`, `γ=7`), Adami boundaries and serial numerical backend match the long study.
The default fluid integrator is adaptive `RDPK3SpFSAL49`, with `reltol=1e-6`,
`abstol=1e-8`. Its `dt=1e-8` is only the initial guess, not a step-size limit.
For the combined case, its largest accepted step is `1.33830577e-4 s`.

The original structure integrator is fixed-step `CarpenterKennedy2N54` at CFL `1.6`,
with nominal substep approximately `1.012e-5 s`. Coupling is between full fluid steps.
Fluid timestep caps are applied through `dtmax`, independently of the structural CFL.

The following entries are **last accepted physical times in seconds**, rather than
exact failing RK-stage times. Every entry terminates with a numerical failure.
The raw and Shepard-only runs target `1.0 s`; the combined screening runs target
`0.1 s`, beyond their observed failures.

| Correction | Default adaptive | Cap `1e-4 s` | Cap `1e-5 s` | Finest tested cap |
|:-----------|-----------------:|-------------:|-------------:|:------------------|
| None | 0.616885 | 0.802027 | 0.697072 | `5e-6 s`: 0.561740 |
| Shepard only | 0.304666 | 0.647438 | 0.539945 | `5e-6 s`: 0.595445 |
| Shepard + mixed gradient | 0.065946 | 0.081949 | 0.073293 | `1e-6 s`: 0.070001 |

The finest caps correspond to nominal acoustic CFL values `c Δt/h` of `0.007071`
and `0.001414`. The combined cap is about 134 times smaller than the largest
accepted step in its default adaptive run. Failure time does not improve monotonically.

All three finest runs fail on grid bounds. Their last accepted maximum fluid heights
are approximately `3.096 m`, compared with the initial column top at `2.0 m`.
Fluid particles also penetrate beyond the nominal side walls. The last maximum speeds
are `17.0`, `45.7`, and `77.7 m/s` for none, Shepard, and combined, respectively.

### Iteration budgets must scale with timestep refinement

The first fine-step attempts hit the default `maxiters=100000`. Those exits are
iteration-budget limits, not evidence of numerical instability. They are retained
with the explicit outcome `iteration_budget` in the outcome table.

Fresh repeats use `maxiters=1000000` and reproduce the preliminary histories exactly
on their shared samples. The final raw and Shepard `5e-6 s` runs take `112726` and
`119558` accepted steps, with **zero rejected steps**, before their grid-bound errors.
The final combined `1e-6 s` run takes `89492` accepted and `18438` rejected steps
before its grid-bound error. These final outcomes are not iteration-budget exits.

Very small accepted-step minima can also arise from sampling-stop remainders;
a minimum step alone is not used to diagnose instability.

## Additional integration controls

These controls use the combined correction at `n=3`:

| Change | Last accepted time [s] | Terminal outcome |
|:-------|-----------------------:|:-----------------|
| Tighter tolerances: `reltol=1e-8`, `abstol=1e-10` | 0.069004 | Timestep collapse |
| Four-times-smaller structural substeps: CFL `0.4` | 0.061862 | Inverse-EOS domain error |
| Fully coupled integration, without splitting, fluid cap `1e-5 s` | 0.078134 | Inverse-EOS domain error |
| Fixed-step Carpenter–Kennedy, step-level coupling, `dt=1e-5 s` | 0.061450 | Inverse-EOS domain error |
| Same Carpenter–Kennedy method and step, stage-level coupling | 0.065150 | Inverse-EOS domain error |

The original RDPK method is incompatible with the split callback's monotonic-stage
requirement. That rejected method combination takes no accepted steps and is recorded
separately. The matched Carpenter–Kennedy pair provides the stage-coupling comparison.

## Evidence independent of timestep size

### Initial free-surface acceleration

A direct RHS evaluation of the nominal hydrostatic initial state gives:

| Resolution | Correction | Maximum acceleration [m/s²] |
|-----------:|:-----------|-----------------------------:|
| 3 | None | 9084.24 |
| 3 | Shepard | 9049.02 |
| 3 | Shepard + mixed gradient | 14947.07 |
| 5 | Shepard + mixed gradient | 29894.29 |

The acceleration peaks at the free surface. Its magnitude increases approximately
twofold when the spacing is halved in the combined case. Initial minimum fluid
pressure is about `-263130 Pa` with Shepard filtering. These are properties of the
spatial RHS at time zero; smaller timesteps do not remove the initial force imbalance.
The density/quadrature and support-truncation mechanisms are described in
[SUMMATION_DIAGNOSIS.md](SUMMATION_DIAGNOSIS.md).

### A frozen EOS-boundary domain violation

A separate stationary one-fluid/one-boundary-particle diagnostic takes **zero time
steps**. Both particles lie within kernel support and have zero velocity. With
summation density, the rarefied fluid has density `278.576566 kg/m³` and pressure
`-357096.357 Pa`. Adami's gravity extrapolation gives boundary pressure
`-357260.327 Pa`, below the inverse Cole EOS limit `-357142.857 Pa`.
The RHS raises `DomainError` at `t=0`, with and without Shepard filtering.

This is a model-domain diagnostic, not the benchmark's initial state. It demonstrates
that once such low-density configurations arise, timestep refinement cannot extend
the real pressure domain of the boundary EOS.

## Interpretation and provenance

The evidence supports treating the initial summation-density/free-surface force
imbalance and low-density boundary-pressure extrapolation as the next issues to
address. Time integration affects the subsequent trajectory, but the tested step-size,
tolerance, subcycling and coupling changes do not establish a stable hydrostatic run.
These finite tests do not exclude every possible timestep or integration method.

Runs use Julia `1.11.9`. All simulation source hashes remain
`8a2a990ffa9424a04739aa14abb6d3e32f958e2bfd1d678a41381f45510bbb89`.
The instrumented adaptive controls exactly reproduce the previous raw, Shepard and
combined pressure/density and plate-deflection histories on shared samples. The
iteration-budget repeats also match their preliminary histories exactly.

The audit retains 21 attempts: 17 numerical failures, three preliminary
iteration-budget exits, and one unsupported method combination. No jobs remain active.
Artifacts are under `out/hydrostatic_corrections/`:

- `wcsph_timestep_sweep/`: primary controls, `initial_acceleration.json`,
  `static_eos_domain.json`, and the combined `timestep_study.csv`/`.json` outcome tables.
- `wcsph_timestep_coupling/`: matched fixed-step step/stage-coupling pair.
- `wcsph_timestep_density_controls/`: raw/Shepard adaptive and capped controls.
- `wcsph_timestep_refinement/`: preliminary default-budget fine-step attempts.
- `wcsph_timestep_budget_refinement/`: definitive fine-step repeats with adequate budgets.

Each job retains its input, accepted-step trace, postprocessing history, log and terminal
diagnostic. Reproduction and analysis scripts are archived in
`wcsph_timestep_sweep/reproduction/`. From the checkout root, using an environment
with the validation dependencies, the definitive fine-step profile can be rerun into
a fresh directory with:

```sh
env GKSwstype=100 OPENBLAS_NUM_THREADS=1 \
  TIMESTEP_SWEEP_VARIANTS=budget_refine \
  TIMESTEP_SWEEP_ROOT=out/hydrostatic_corrections/timestep_repeat \
  julia --project=/path/to/validation/environment --threads=1 \
  out/hydrostatic_corrections/wcsph_timestep_sweep/reproduction/run_wcsph_timestep_sweep.jl
```

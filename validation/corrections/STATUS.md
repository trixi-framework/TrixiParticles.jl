# Correction validation: current status and limitations

This is the status of the validation work in
[PR #1286](https://github.com/trixi-framework/TrixiParticles.jl/pull/1286).
Passing implementation/regression checks and finishing an integration are distinct
from demonstrating hydrostatic accuracy or full-discretization convergence.

## What the validation establishes

- The manufactured study records 404 measurements at `N=12,24,48,96`, separating
  truncated-boundary and symmetric-interior behavior. It checks local interpolation,
  density, gradient, pressure and constant-pressure responses.
- The coupled runner supports WCSPH/EDAC, continuity/summation density, the ten
  Shepard/gradient combinations, evolved-volume density resets, and declaration-order
  controls. It retains failed and blocked cases in the outcome ledger.
- The reference-initialized WCSPH setup preserves its prescribed density and EOS
  pressure at startup, with all particle masses retained. Static checks cover
  `n=3,5,9,17`. This is the explicit opt-in reference-offset formulation
  `ρ(t)=ρ(0)+R(t)-R(0)`, not an assertion that the unmodified raw sum reproduces the
  prescribed hydrostatic density.
- Action/reaction, callback state handling, EOS consistency, initial-state response
  to motion, and restart references have focused regression coverage.

The validation branch contains the stack-compatible support changes needed by these
checks. The boundary-density history fix is isolated in
[PR #1368](https://github.com/trixi-framework/TrixiParticles.jl/pull/1368); the WCSPH
reconstructed-boundary pressure-force fix is isolated on `main` in
[PR #1369](https://github.com/trixi-framework/TrixiParticles.jl/pull/1369).
These fixes do not by themselves establish stability of every correction variant.

## Current shortcomings

### The coupled correction matrix is not universally stable

The archived four-level, four-block study has 176 planned outcomes: 108 trajectories
reach `1.0 s`, 60 fail, and eight reversed-order controls are blocked. This archive
predates the later WCSPH predictor, reference-initialization and boundary-force fixes.
Its numerical audit also retains ten bound exceedances; it is not an all-passed study.

After reference initialization and the WCSPH boundary-force fix, the matched,
unclipped summation-density rerun gives:

| Plate resolution | Reached `1.0 s` | Later failure | Blocked reversed control |
|-----------------:|---------------:|--------------:|-------------------------:|
| 3 | 5 | 5 | 1 |
| 5 | 3 | 7 | 1 |

All 20 forward initial states are consistent to roundoff, but only eight of the 22
planned outcomes reach the horizon. The later failures include grid-bound errors,
inverse-EOS domain errors, and timestep collapse. The full four-level cross-product
has **not** been repeated with all the latest fixes.

### Smaller timesteps and rigidifying the plate do not remove the tested runaway

Fixed-plate controls also develop large fluid velocities. A boundary-free periodic
fluid test has tensile modes with growth rates around `31.35 s⁻¹` at `-10000 Pa`.
The measured growth over `0.1 s` is `22.9803` at both `10 µs` and `5 µs` timesteps,
matching the spatially linearized prediction. This identifies a semidiscrete growing
mode in the tested state, rather than a timestep-stability failure.

### Stabilized execution does not establish pressure accuracy

Separate clipped-pressure `n=3` kernel/mixed/combined controls all reach `1.0 s`, but
their late pressure errors remain about `9–12%`. A density-floor diagnostic does not
improve them and increases the combined-case error to about `16.4%`. The density floor
is retained as a negative control and is not part of the production implementation.
These controls use different settings from the matched unclipped study.

### Plate under-resolution and order sensitivity remain visible

The isolated TLSPH control reproduces the coarse structural deflection bias:
approximately `+2.44%`, `-15.33%`, `-13.25%`, and `-7.77%` at `n=3,5,9,17` against
the benchmark reference. The good `n=3` agreement is not evidence of structural
resolution. The archived coupled study also retains declaration-order bound
exceedances and late pressure peaks. Local affine consistency and interface momentum
balance do not imply full-solver convergence or a stationary late-time solution.

## Protocol and reference provenance

Current output records the density model, pressure clipping, viscosity, integration
settings, density-boundary prediction, initialization policy and reconstructed-pressure
gradient policy. Reuse and inter-level comparisons reject incompatible policies.

The checked-in `validation_reference_long*` and `validation_reference_summation*`
files are labelled measurements from earlier protocols. Compatible EDAC and
continuity-control initial states remain fixtures; current reference-initialized
WCSPH summation is checked against the analytical initial state. Archived observations
are not regeneration or accuracy targets for changed defaults.

Full histories, HTML/PDF reports, per-job logs and one-off diagnostic snapshots are
generated artifacts under ignored output directories. The reusable runners, report
builder, compact reference summaries and diagnostic findings are kept in source.
The report preserves missing/failed outcomes and bounds rather than substituting zero
errors or widening acceptance thresholds.

## Details and reproduction

- [Manufactured operators and report workflow](README.md)
- [Coupled runner, parameters, refinement and archived results](../hydrostatic_water_column_2d/README.md)
- [Initial-state formulation](../hydrostatic_water_column_2d/INITIAL_STATE.md)
- [Summation-density diagnosis](../hydrostatic_water_column_2d/SUMMATION_DIAGNOSIS.md)
- [Timestep controls](../hydrostatic_water_column_2d/TIMESTEP_DIAGNOSIS.md)
- [Boundary-force, fixed-plate, tensile-mode and stabilization controls](../hydrostatic_water_column_2d/POST_STARTUP_STABILITY.md)

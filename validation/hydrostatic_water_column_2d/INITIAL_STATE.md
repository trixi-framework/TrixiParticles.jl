# Reference-consistent hydrostatic initial state

The hydrostatic plate example now preserves its prescribed initial density when
WCSPH is run with summation density. It prepares the exact compressible Cole
hydrostatic profile and enables `initial_density_correction=true` on the fluid system.
The example option `initialize_summation_density=false` reproduces the earlier setup.

## Density operator reference

If `R_a` denotes the configured raw or Shepard-filtered density operator, the
reference-initialized density is

```text
ρ_a(t) = ρ_a(0) + R_a(t) - R_a(0).
```

The reference error is measured once from the actual initial configuration and
interacting boundaries during `semidiscretize`. Subsequent evaluations still respond
to particle motion. A rejected trial evaluation does not change the reference.
The measurement uses density-only startup phases, so an uncorrected temporary
density does not enter the final EOS boundary update before its offset is known.

For raw summation, this is a constant offset to `Σ m_b W_ab`. With Shepard filtering,
the filter continues to use raw summation volumes. Its configured filtered operator
gets a separate constant offset. EOS-backed boundaries are predicted using the
unfiltered reference-corrected density estimate, then refreshed from the final
physical density and pressure after filtering.

All fluid masses, solid material masses, and dummy-boundary hydrodynamic masses are
retained. Initialization uses neighbor passes and per-particle vectors, rather than
a global mass-fitting system. This avoids the positivity and mass-preservation
conflict encountered by a mass-only calibration at finer resolutions.

## Hydrostatic profile

The free surface is half a spacing above the highest translated fluid particle.
For Cole exponent `γ != 1`, the profile is

```text
ρ(y) = ρ₀ [1 + (γ-1) g max(H-y,0) / c²]^(1/(γ-1)),
p(y) = Cole(ρ(y)).
```

For `γ=1`, the density ratio is `exp(g max(H-y,0)/c²)`. Density and EOS pressure
are prepared consistently before building the WCSPH system; particle masses are
not recomputed from the adjusted profile.

Static checks at `n=3,5,9,17` reproduce this initial density and EOS pressure to
floating-point roundoff. Regression checks verify unchanged masses, response to
particle motion, restored trial states, declaration-order changes, and retention
of reference data through in-memory and VTK restart.

The correction enforces initial density/EOS consistency. Remaining discrete momentum
residuals, later stability, and late-time accuracy are measured by the validation
runner rather than set to zero by the initializer.

## Provenance

New case records use `wcsph_summation_initialization="operator-offset-hydrostatic"`
and include initial density, pressure, reference-offset, and total-fluid-mass
diagnostics. Cached trajectories and failures from earlier initialization policies
are excluded from reuse for WCSPH summation-density jobs. Refinement comparisons
keep these policies separate.

The boundary-history and timestep investigations describe earlier initial states:
[SUMMATION_DIAGNOSIS.md](SUMMATION_DIAGNOSIS.md) and
[TIMESTEP_DIAGNOSIS.md](TIMESTEP_DIAGNOSIS.md). Their measured outputs are retained
as archived controls.

Fresh coarse-resolution results are written to
`out/hydrostatic_corrections/wcsph_operator_initialized_refinement/`. Per-case initial
records and logs accompany the outcome ledger, with `initialization_comparison.csv`
and `initialization_audit.json` providing the before/after checks.

## Coarse matched rerun

This rerun precedes the subsequent [WCSPH pressure-boundary flux fix](POST_STARTUP_STABILITY.md).

All 20 forward initial states have zero density/EOS residual in the recorded
diagnostics. The largest relative hydrostatic pressure error is `1.56e-14`.
The original adaptive solver, unclipped pressures, and zero added artificial viscosity
are retained for the `1.0 s` rerun.

| Resolution | Reached `1.0 s` | Later numerical failure | Blocked reversed control |
|-----------:|---------------:|------------------------:|-------------------------:|
| 3 | 5 | 5 | 1 |
| 5 | 3 | 7 | 1 |

Before reference initialization, none of these summation-density trajectories reached
the horizon. The new initial state removes that startup inconsistency, but later
stability and accuracy remain unresolved for several correction variants. The twelve
later failures comprise five grid-bound errors, three inverse-EOS domain errors, and
four timestep collapses; they are retained rather than interpreted as successful runs.

The measured run source hash is pinned in the audit. A subsequent density-only startup
safeguard avoids evaluating invalid temporary EOS boundary states. Its reference arrays,
initial density/pressure, and complete initial RHS are verified to be bitwise identical
to the measured startup for all 20 coarse configurations; this equivalence audit is
saved alongside the outcomes.

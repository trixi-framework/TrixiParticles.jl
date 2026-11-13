# WCSPH post-startup pressure-force investigation

These checks follow [reference-consistent initialization](INITIAL_STATE.md). All
controls start with the prescribed hydrostatic density and EOS pressure. They isolate
the remaining wall-flux and fluid-instability mechanisms.

## Reconstructed-pressure boundary flux

Adami/Bernoulli dummy pressure is reconstructed from the fluid, rather than evolved
as an independent pressure degree of freedom. Applying the generic two-endpoint
asymmetric-gradient pressure formula also consumes a ghost-side correction whose
support differs from the fluid support.

A closed, stationary, uniform-density patch at constant pressure `10000 Pa` has exact
fluid acceleration zero. The old WCSPH routing produces accelerations up to
`1.1427018 m/s²`; a fluid-side gradient closure reduces the maximum across the
diagnostic matrix to `1.41e-13 m/s²`. The matrix covers both density calculators,
their matching pressure forms, the inter-particle-averaged pressure form, and several
gradient/boundary-gradient combinations.

The implementation now uses the fluid-side gradient for the three built-in WCSPH
pressure forms at Adami/Bernoulli dummy boundaries. Fluid-fluid pairs still use the
two-endpoint conservative forms. The structure-fluid path evaluates the same
fluid-oriented pair force and applies its exact opposite.

The focused regression covers walls, rigid bodies, TLSPH bodies, both density
calculators, Adami/Bernoulli reconstruction, and the supported pressure forms:
**76 pass / 30 fail before the fix; 106/106 pass with the fix**.

New results record `wcsph_extrapolated_pressure_gradient="fluid-sided"`. Earlier
WCSPH trajectories and cached failures with the old routing are excluded from reuse,
and refinement comparisons separate this pressure-flux policy.

## Fixed-plate controls

The plate was replaced by stationary dummy particles at exactly the same locations,
with the same hydrodynamic masses, pressure model and interaction matrix. Initial
fluid RHS differences are below `1e-6 m/s²` (approximately roundoff in the measured
controls). This removes structural dynamics and split coupling.

These are diagnostic screens with target `0.7 s`. A run is deliberately terminated
when the accepted maximum fluid speed crosses the reference sound speed `50 m/s`.
Such a termination is recorded as a diagnostic stop, not a successful trajectory or
an integrator exception.

| Correction | Plate | Original closure: threshold time [s] | Fluid-side closure: threshold time [s] |
|:-----------|:------|-------------------------------------:|---------------------------------------:|
| Kernel | Coupled | 0.429499 | 0.379940 |
| Kernel | Fixed | 0.501107 | 0.391081 |
| Shepard + mixed | Coupled | 0.317633 | 0.477865 |
| Shepard + mixed | Fixed | 0.353800 | 0.526030 |

The mixed-gradient-only fluid-side screen reaches `0.7 s` with maximum speed about
`0.149 m/s`. The kernel/combined runaway also occurs with the plate fixed, so elastic
subcycling and partitioned coupling are not its sole cause. The boundary correction
repairs a demonstrated consistency error but is not a universal stability remedy.

## Fluid-only tensile modes

A periodic `6×6` fluid lattice, without walls or structure, was linearized around a
uniform stationary state. The scaled full velocity/position/density Jacobian uses
central differences at two dimensionless perturbation sizes, `1e-6` and `5e-7`.
The study compares raw summation, Shepard-filtered summation, and continuity density
with no gradient correction, kernel correction and mixed correction.

At pressure `-10000 Pa`, the largest positive real eigenvalues are approximately
`42.39 s⁻¹` without gradient correction and `31.35–31.57 s⁻¹` with kernel/mixed
correction. These rates are stable under perturbation-size refinement. The initial
RHS is at roundoff, so the growth is not caused by a failed initial equilibrium.
Zero-pressure rates approach zero with numerical perturbation size. Weak lattice
growth modes also occur at positive pressure in this finite stencil; the analysis
does not establish unconditional stability of a pressure-clipped scheme.

The dominant kernel-correction tensile mode was then integrated from a `1e-6`
scaled perturbation for `0.1 s`, using fixed-step Carpenter–Kennedy integration:

| Timestep [s] | Predicted growth | Measured growth |
|-------------:|-----------------:|----------------:|
| `1e-5` | 22.98031654 | 22.98031709 |
| `5e-6` | 22.98031654 | 22.98031596 |

Halving the timestep leaves the growth unchanged to roughly `5e-8` relative. This
is a growing mode of the spatial discretization in the tested tensile state, rather
than a timestep-stability failure.

## Explicit pressure-clipping control

With the corrected initial state and fluid-side wall flux, three separate `n=3`
controls enable the existing `clip_negative_pressure=true` option. They retain zero
added viscosity and target `1.0 s` with the original adaptive/split solver settings.

| Correction | Outcome | Mean pressure L2 error over `[0.5,1.0]` | Mean deflection error |
|:-----------|:--------|----------------------------------------:|----------------------:|
| Kernel | Success, `1.0 s` | 11.63% | 2.78% |
| Mixed | Success, `1.0 s` | 9.44% | 2.65% |
| Shepard + mixed | Success, `1.0 s` | 11.58% | 2.73% |

Pressure clipping suppresses the tested tensile runaway over this horizon, but these
pressure errors remain substantial. Completion alone does not qualify hydrostatic
accuracy. This control is recorded separately from the unclipped matched study.

### Density-floor volume control

A further diagnostic also bounds the physical density below by `ρ₀` before pressure
and final gradient updates. It tests whether large `m/ρ` weights of under-dense
particles explain the clipped runs' remaining error. This is an in-process diagnostic
overlay, not a change to the production density operator.

| Correction | Outcome | Mean pressure L2 error | Mean deflection error |
|:-----------|:--------|-----------------------:|----------------------:|
| Kernel | Success, `1.0 s` | 11.50% | 2.79% |
| Mixed | Success, `1.0 s` | 9.50% | 2.68% |
| Shepard + mixed | Success, `1.0 s` | 16.43% | 2.82% |

The floor does not resolve pressure accuracy and worsens the combined case. It is
therefore retained as a negative control rather than adopted as a fix.

## Matched unclipped rerun after the boundary fix

The fresh `n=3,5` matrix retains the original adaptive/split solver, unclipped pressure
and zero added viscosity. All forward cases have consistent initial density/EOS
states. The terminal execution ledger is:

| Resolution | Reached `1.0 s` | Later numerical failure | Blocked reversed control |
|-----------:|---------------:|------------------------:|-------------------------:|
| 3 | 5 | 5 | 1 |
| 5 | 3 | 7 | 1 |

The aggregate completion count remains `8/22` planned outcomes. The mixed-gradient
`n=3` case now completes while the Shepard/blended `n=3` case no longer completes;
the ledger preserves both changes. The fix removes a pressure-consistency defect
without establishing universal dynamical stability. The twelve later failures are
seven grid-bound errors, one inverse-EOS domain error and four timestep collapses.

## Verification

- Focused boundary regression: `106/106` passed, after `30` failures before the fix.
- Full correction consistency suite: `900/900` passed.
- Fluid RHS, including structure reactions: `6848/6848` passed.
- Pressure acceleration: `33/33` passed.
- Reference initialization/restart checks: `77/77` passed.
- Hydrostatic validation/refinement checks and repository Aqua checks passed.
- The consolidated audit verifies terminal outcomes, the frozen-field force residual,
  fixed/coupled initial-RHS matching, spectra and timestep-refined mode growth.

## Artifacts

All directories below are under `out/hydrostatic_corrections/`:

- `wcsph_pressure_boundary_controls/`: nine coupled/fixed-plate diagnostic screens,
  accepted-state traces and serialized accepted/working endpoint states.
- `wcsph_fluid_side_boundary_refinement/`: the fresh unclipped coarse correction matrix.
- `wcsph_post_init_clipping_controls_v2/`: completed clipping controls with terminal
  solver diagnostics and physical histories. An earlier attempt stopped in auxiliary
  postprocessing and is retained separately.
- `wcsph_post_init_density_floor_controls/`: a separate diagnostic of bounded-density
  volume weights; its results are not part of the matched unclipped study.

The frozen-field probe, periodic spectra, timestep-refined mode-growth check and
consolidated `boundary_pressure_audit.json` are archived with the fresh boundary-refinement
study, alongside `boundary_flux_comparison.csv` and reproduction scripts.

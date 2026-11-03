# Hydrostatic water column on an elastic plate

This benchmark follows the elastic plate under a hydrostatic water column in
J. O'Connor and B. D. Rogers, *A fluid–structure interaction model for free-surface
flows and flexible structures using smoothed particle hydrodynamics on a GPU*,
Journal of Fluids and Structures (2021),
https://doi.org/10.1016/j.jfluidstructs.2021.103312.

## Files

- `../../examples/fsi/hydrostatic_water_column_2d.jl`: reusable simulation setup.
- `validation.jl`: original WCSPH/EDAC plate-deflection validation.
- `plot.jl`: comparison of original validation runs, stored reference histories, and
  the analytical plate deflection.
- `validation_reference_{wcsph,edac}_{3,5,11,13}.json`: original reference histories.
- `validation_corrections.jl`: correction-enabled WCSPH/EDAC refinement study.
- `validation_long.jl`: paper-length (`1.0 s`) Shepard/gradient correction matrix
  within the default density formulation of each scheme,
  with a `[0.5, 1.0] s` averaging window and no VTU output by default.
- `correction_methods.jl`: shared method definitions used by the runner and plot axes.
- `validation_refinement.jl`, `refinement_util.jl`, and `refinement_worker.jl`: parallel,
  resumable execution of the full scheme × density-formulation cross-product at four levels,
  with verified reuse of completed trajectories and terminal failure outcomes.
- `refinement_metrics.jl` and `refinement_report.jl`: matched inter-level errors and
  automatic report generation after a complete campaign.
- `validation_reference_long{,_order}.csv` and `validation_reference_long_parameters.json`:
  eight-significant-digit summaries of all 46 long runs, including observed bound exceedances.
- `validation_summation_density.jl`: matched four-way `SummationDensity` matrix in both schemes,
  with continuity/reinitialization controls.
- `validation_structure.jl` and `structure_util.jl`: isolated, prescribed-load TLSPH
  equilibrium/refinement control, without fluid evolution or time integration.
- `validation_reference_summation_{density,order}.csv` and
  `validation_reference_summation_parameters.json`: measured resolution-3/5 reference
  summaries and their reproducible configuration (eight significant digits).
- `correction_util.jl`: hydrostatic-error, plate-deflection, interface-force, and
  system-order diagnostics for the correction study.
- `plot_correction_results.jl`: self-contained visual report of correction-study histories.

The analytical central deflection is

```text
D = E * thickness^3 / (12 * (1 - ν^2))
deflection = -0.0026 * g * (ρ_fluid * water_height + ρ_structure * thickness) / D
```

The original `validation.jl` averages the numerical deflection over `[0.25, 0.5]`
and compares it with this value. Run it with `include` or use `trixi_include` to
change `n_particles_plate_y` and `tspan`.

## Correction study

WCSPH summation-density cases now use [reference-consistent initialization](INITIAL_STATE.md)
to preserve the prescribed hydrostatic density and EOS pressure without changing particle masses.
The terminal study and timestep diagnostics below archive the earlier initialization;
new outputs identify the initialization policy explicitly.
The [post-startup investigation](POST_STARTUP_STABILITY.md) documents the subsequent
WCSPH boundary-flux correction, fixed-plate controls and fluid-only tensile modes.
The concise [validation status](../corrections/STATUS.md) lists the current shortcomings
and distinguishes the measured protocol versions.

The follow-up [WCSPH summation diagnosis](SUMMATION_DIAGNOSIS.md) identifies the
initial density/EOS mismatch and the corrected Shepard/EOS-boundary history dependence.
The terminal 176-outcome report is a pre-predictor archive; affected results are not
eligible for post-fix reuse. The archived
`validation_reference_summation_predictor.csv` supplies compatible EDAC and
continuity-control initial-state fixtures. Reference-initialized WCSPH summation
cases are checked against the analytical initial state instead.

The correction study uses the same geometry and material parameters: a `1 × 2`
water column, a `1 × 0.05` plate, fluid density `1000`, structure density `2700`,
gravity `9.81`, `E = 67.5e9`, `ν = 0.3`, and artificial sound speed `50`.
All cases use Wendland C2 kernels, `h / Δx = √2`, serial neighborhood updates,
the serial execution backend, and the example's split fluid/structure integration.
The default-density branch retains the original example's models: continuity density with
Molteni–Colagrossi diffusion (`δ = 0.1`) for WCSPH and summation density for EDAC.
Neither pressure clipping nor additional artificial viscosity is enabled. Each
baseline/corrected pair has the same physical and numerical settings. The full
cross-product additionally exercises the other density formulation in each scheme.

### Archived terminal matched-study coverage

The earlier four-level scheme × density study finished all scheduled work. Of its 176
requested trajectories/controls, 108 reach `1.0 s`, 60 fail, and eight reversed-order
controls are blocked by failed forward runs. No jobs remain running or pending.

| Scheme | Density formulation | Reached `1.0 s` | Failed | Blocked |
|:-------|:--------------------|-----------------:|-------:|--------:|
| WCSPH | Continuity | 44 | 0 | 0 |
| WCSPH | Summation | 0 | 40 | 4 |
| EDAC | Summation | 34 | 8 | 2 |
| EDAC | Continuity | 30 | 12 | 2 |

These are archived execution outcomes, not universal accuracy acceptance. That
unclipped WCSPH-summation configuration had large startup defects and no trajectory
that reaches the requested horizon. Numerical order-bound exceedances and EDAC
instabilities remain in the report and audit; failed or blocked cases have no
invented late-time statistics. The final outcome ledger, measured histories,
initialization records and diagnostic controls are embedded in the report.

Each scheme/resolution runs the full **2 × 5 matrix**: Shepard density correction
off/on crossed with five fluid/boundary gradient corrections. The case identifiers
and plot-axis definitions share one source (`correction_methods.jl`):

Here **2 × 5** counts Shepard off/on and gradient choices; it does not count
continuity versus summation density. The earlier long-study density coverage was:

| Scheme | Continuity density | Summation density |
|:-------|:-------------------|:------------------|
| WCSPH | Ten correction combinations through `1.0 s`, all four resolutions | Four-way short study through `0.02 s`, resolutions 3 and 5, with clipping and added viscosity |
| EDAC | One mixed-gradient + periodic-reinitialization control per resolution | Ten correction combinations through `1.0 s`, all four resolutions |

The extended `validation_refinement.jl` now requests all four blocks, including full
long-duration correction matrices for **EDAC continuity** and **WCSPH summation**.
Both density choices are supported independently by the shared runner through
`wcsph_density_mode` and `edac_density_mode`. The stabilized short summation study has
different settings and is not substituted for these matched long-study cases.

| Gradient correction | Shepard off | Shepard on |
|:--------------------|:------------|:-----------|
| None | `none` | `shepard` |
| Kernel normalization (`KernelCorrection`) | `kernel` | `shepard_kernel` |
| First-moment matrix (`GradientCorrection`) | `gradient` | `shepard_gradient` |
| Blend with uncorrected gradient, `λ=0.5` | `blended` | `shepard_blended` |
| Kernel normalization + first-moment matrix | `mixed` | `combined` |

**Shepard on has density-model-specific meaning:** continuity density uses evolved-volume
reinitialization every `0.01 s`; WCSPH then updates EOS pressure and EDAC preserves its
independently evolved pressure. Summation density uses a Shepard filter at every
RHS evaluation in either scheme. Axis labels
explicitly identify the base density model and both correction choices.

An additional EDAC `continuity_reinit` control uses continuity density, prescribed
Shepard resets, and mixed gradients. Its WCSPH counterpart is already `combined`.
The combined case is also repeated with reversed system declaration order.
The earlier default-density campaign has **92 outcomes** at `(3, 5, 9, 17)`, including
four EDAC continuity controls. The full four-block campaign has **176 outcomes**:
160 forward correction cases and 16 combined-case order repeats. Its explicit
continuity blocks make the old `continuity_reinit` alias redundant, so the full plan
uses the ten `MATRIX_CASES` without duplicate controls.

Compare each gradient variant with `none`, or each Shepard+gradient variant with
`shepard`, to isolate the gradient correction. The earlier uncorrected WCSPH
summation setup overwrote hydrostatic density/pressure before the first step.
Reference initialization now removes this startup confound; later stability and
accuracy remain separate validation questions.
Fluid and boundary correction options are named independently so `trixi_include`
does not rewrite a boundary keyword when changing the fluid correction.

Boundary gradients are corrected on both the stationary wall and the plate's
hydrodynamic boundary model. Adami pressure extrapolation continues to determine
the boundary pressure/density; no Shepard density overwrite is applied to it.
The plate's structural kernel correction remains the TLSPH correction.
For EDAC's default inter-particle pressure law, the Adami boundary flux uses the
fluid's corrected gradient: ghost pressure is reconstructed, not an independent
pressure DOF. The boundary correction does not supply a second pressure gradient.
The structure still receives the exact opposite fluid-oriented pair force.
The built-in WCSPH pressure forms likewise use the fluid-side gradient at
Adami/Bernoulli dummy boundaries, as recorded by
`wcsph_extrapolated_pressure_gradient="fluid-sided"`.

The `combined` case is repeated with the fluid and structure declaration order
reversed, replaying the forward accepted fluid time grid so the comparison uses
the same time discretization rather than two different adaptive/split trajectories.
The continuity case reinitializes density from the **evolved** state,
starting after the initial time, with `DensityReinitializationCallback`.
Its update logic is scheduled at prescribed physical times with `PresetTimeCallback`
so adaptive-step or declaration-order differences do not change the filtering times.

### Running

Use an environment containing `TrixiParticles`, `OrdinaryDiffEqLowStorageRK`,
`CSV`, `DataFrames`, and `JSON` (the project's test environment contains these).

```julia
using TrixiParticles
include(joinpath(validation_dir(), "hydrostatic_water_column_2d",
                 "validation_corrections.jl"))
```

Defaults are `resolutions=(3, 5, 9, 17)`, both schemes, `tspan=(0.0, 1.0)`, sampling every
`0.0025`, density reinitialization every `0.01`, and VTU output enabled. The
resolution is the number of particles across the plate thickness, so
`Δx = 0.05 / (resolution - 1)`. Late statistics use `[0.5, 1.0] s`.
For the same full study with sampling every `0.005 s` and VTU output disabled:

```julia
trixi_include(@__MODULE__,
              joinpath(validation_dir(), "hydrostatic_water_column_2d", "validation_long.jl"))
```

This runs to the paper's `1.0 s` horizon, rather than using startup-only `0.02 s`
controls or the earlier `0.35 s` raw/mixed subset. Both schemes and all four resolutions
have identical durations and complete matrix coverage. Late mean density/pressure
errors, plate-deflection span and the difference between the two half-window means
are recorded; a long time span alone is not treated as proof of temporal stationarity.

### Four-level refinement and resumption

The sequence `(3, 5, 9, 17)` halves `Δx = 0.05/(n-1)` at every level, including the
kernel smoothing length. It gives three adjacent refinement comparisons per method.

| Plate layers `n` | `Δx` [m] | Fluid particles |
|----------------:|---------:|----------------:|
| 3 | `0.025` | 3,200 |
| 5 | `0.0125` | 12,800 |
| 9 | `0.00625` | 51,200 |
| 17 | `0.003125` | 204,800 |

For parallel execution, use an environment containing the simulation dependencies,
`SHA`, and `Plots` for the final report:

```julia
trixi_include(@__MODULE__,
              joinpath(validation_dir(), "hydrostatic_water_column_2d",
                       "validation_refinement.jl");
              workers=12, output_root="out_hydrostatic_refinement",
              reuse_directories=("path/to/completed/study",),
              operator_file="out/correction_convergence.csv")
```

This schedules **160 forward matrix cases and 16 order repeats** across both schemes,
both density formulations and four resolutions. `density_modes=(:continuity, :summation)`
is the default; a subset can be selected explicitly. Density formulation is included
in job IDs, campaign directories, outcome ledgers and comparison keys. The original
baseline settings remain unclipped and have no added artificial viscosity; the WCSPH
continuity model retains Molteni–Colagrossi diffusion (`δ=0.1`).
Each worker is a separate, single-threaded Julia process using the serial numerical
backend. The combined forward/reversed pair shares one worker so it can replay the
forward accepted time grid. Per-job `run.log`, input and completion records live under
`jobs/<scheme>_<case>_<n>/`. `refinement_progress.json` records pending, running,
completed, reused and failed jobs. Compatible completed cases are reused on restart;
settings, time coverage and simulation-source hashes are checked. Imported campaigns
record that their original source hash is unavailable rather than claiming a new one.
New workers back up the sampled histories every four postprocessing calls, preserving
progress and pre-failure traces without changing sampling times or the numerical model.

Run the same command after an interruption to resume. Completed results are preserved,
and incomplete jobs are retried. A successfully finished numerical trajectory is a
completed run; pressure accuracy and order-bound acceptance remain separate diagnostics.
When all cases finish, `campaigns/<scheme>_<n>/` contains the assembled histories and
summaries. The coordinator writes the combined summary, inter-level CSV and HTML report;
`completion_command` can run an additional PDF/browser export command.
`refinement_audit.json` checks plan coverage and records numerical acceptance separately;
`hydrostatic_acceptance.csv` retains every checked residual and unchanged bound, including
failures. A completed campaign can therefore still have numerical acceptance failures.

Available results can also be published while the campaign is running or after a
worker failure:

```sh
julia --project=your-validation-environment \
  validation/hydrostatic_water_column_2d/publish_refinement.jl out_hydrostatic_refinement
```

An optional second argument names a directory of tagged diagnostic subdirectories.
Their streamed histories, failure-state snapshots, archived logs and summaries are
embedded as supplementary downloads, without entering the primary method statistics.

The report records every planned case in `hydrostatic_case_status.csv`. Failed forward
trajectories appear as **FAIL** cells, running/pending work as **RUN/WAIT**, and a reversed
control blocked by a failed forward run is explicitly marked `blocked` in the ledger.
Missing results have no numeric error value. Validated forward trajectories can be
included while their paired reversed job is unfinished, so report trajectory counts
may exceed the completed-pair count in the worker progress summary. Failed controls
are unavailable diagnostics, not zero residuals. Error ranges in panel G compare only
the same methods completed at every displayed level; failed or pending methods cannot
make the fine-grid envelope look artificially better.
Instrumented or alternative-grid investigations carry `diagnostic_variant.json`;
the primary campaign refuses to reuse these runs, and plotting keeps their protocols
separate from the matched study.

For a concise progress summary:

```julia
include(joinpath(validation_dir(), "hydrostatic_water_column_2d", "refinement_util.jl"))
HydrostaticRefinement.refinement_status("out_hydrostatic_refinement")
```

The existing `validation_reference_long*.csv/json` files record the completed
two-level `(3, 5)` campaign. The terminal four-level cross-product results and full
outcome ledger are written under the refinement output directory and embedded in its report.
The isolated structural control has also been evaluated on the h-halving sequence:
signed biases are `+2.437%`, `-15.331%`, `-13.254%`, and `-7.765%` at `n=3,5,9,17`.

For a short run matching the CI regression:

```julia
trixi_include(@__MODULE__,
              joinpath(validation_dir(), "hydrostatic_water_column_2d",
                       "validation_corrections.jl");
              resolutions=(3,), tspan=(0.0, 0.002), sample_dt=0.001,
              correction_cases=(:none, :shepard, :mixed, :combined, :continuity_reinit),
              reinitialization_dt=0.0005, save_solution=false,
              output_root="out_hydrostatic_corrections/short_run")
```

Select cases with `methods=(:edac,)` and `correction_cases=(:combined,)`, and
disable the ordering repeat with `check_reverse_order=false`.

### Explicit summation-density matrix

The PR-thread request for matched summation-density cases is provided by:

```julia
trixi_include(@__MODULE__,
              joinpath(validation_dir(), "hydrostatic_water_column_2d",
                       "validation_summation_density.jl"))
```

This runs `none`, `shepard`, `mixed`, and `combined` with `SummationDensity()` in
**both** WCSPH and EDAC, plus one continuity-density reinitialization control per
scheme and a reversed declaration-order repeat. Defaults are resolutions `(3, 5)`,
`tspan=(0.0, 0.02)`, `sample_dt=0.0025`, and `reinitialization_dt=0.005`: 24 runs.
The output root is `out_hydrostatic_summation_corrections`.

All four WCSPH summation cases share pressure clipping and Monaghan artificial
viscosity (`α=0.1`, `β=0`) as explicit stabilization controls; density
diffusion is disabled for summation density. EDAC retains its pressure-diffusion
damping and unclipped pressure. The physical parameters, kernel, spacing, and numerical
settings are identical across the four correction choices within each scheme.

These are **short density-consistency/stabilization cases**, with a different protocol
from the unclipped long study. Their archived references retain the earlier startup
error; current WCSPH cases preserve the analytical initial density/EOS state.
The continuity-density WCSPH study supplies an additional physical baseline.
Short runs have no steady deflection-error value when they do not reach the averaging
window, and reaching a short horizon does not establish hydrostatic accuracy.

The shared runner also exposes `wcsph_density_mode=:continuity` or `:summation`,
`wcsph_pressure_clipping`, and `wcsph_artificial_viscosity`. Every actual case records
its density calculator, clipping, viscosity, and diffusion choices in `parameters.json`.

### Diagnostics and output

The default output root is `out_hydrostatic_corrections`. Each case has a directory
`<method>_<correction>_<resolution>[_reversed]` containing a
`validation_run_hydrostatic_corrections.json` history, its CSV equivalent, and
optional VTU files. JSON metadata records the scheme, corrections, solver, and
software versions. The root also contains:

- `parameters.json`: physical parameters, resolutions, time intervals, and selected cases;
- `hydrostatic_corrections.csv`: final/maximum density and pressure errors, equation-of-state
  residual, interface force residual and scale, final deflection, averaged deflection
  error, last density-reinitialization time, and initial-state momentum/affine-gradient residuals;
- `hydrostatic_correction_order.csv`: differences between forward/reversed mean-density,
  mean-pressure, and plate-deflection histories, normalized by reference density,
  hydrostatic pressure scale, and analytical deflection respectively;
- `hydrostatic_correction_comparisons.csv`: matched `mixed/none` and `combined/shepard`
  error ratios. Ratios above one indicate deterioration and are retained.

The summaries additionally record initial density/pressure error, the reinitialization
count, evolved-volume normalization residual, separation from an initial-fluid-volume
counterfactual, pre-reset density evolution, immediate post-reset WCSPH EOS residual,
and EDAC pressure change caused by the reset. `late_density_error` and
`late_pressure_error` are means over the recorded averaging window;
`late_deflection_span` is its peak-to-peak deflection divided by the analytical
deflection magnitude, and `late_deflection_drift` is the normalized absolute
difference between the means of its two halves. They are missing for short runs
that do not reach the window. Every case records its density and gradient method
definition, and campaign settings include the complete method dictionary and versions.

Completed cases are saved after each run. Density and pressure errors are relative
discrete L2 norms over all fluid particles. The initial free surface is
`H = maximum(initial_fluid_coordinates[2, :]) + Δx/2`, including the tank's coordinate
translation. For EDAC, `p(y) = ρ₀g max(H-y, 0)` and density is constant `ρ₀`.
For WCSPH, both reference fields solve `dp/d(depth) = ρ(p)g` with the Cole EOS:

```text
ρ(depth) = ρ₀ * (1 + (γ-1) * g * depth / c²)^(1/(γ-1))
p(depth) = ρ₀ * c² / γ * ((ρ(depth)/ρ₀)^γ - 1)
```

The reference helper also handles `γ = 1` through the exponential limit.
Histories store mean density and pressure. WCSPH additionally records the maximum pointwise discrepancy
between stored pressure and the equation of state evaluated at the updated density,
normalized by `ρ₀ * g * H`. This diagnostic is absent for EDAC, whose pressure is
an independent evolved variable.

The interface diagnostic independently assembles the production fluid-to-plate
and plate-to-fluid interactions. It records the two total force vectors, the
hydrodynamic reaction on clamped plate particles, and
`norm(F_fluid + F_plate) / (norm(F_fluid) + norm(F_plate))`. Material masses weight
the plate accelerations. The clamped contribution is included in `F_plate`.
This is an **interface action/reaction check**; gravity, stationary-wall forces,
and elastic/clamp forces are not part of this diagnostic.

The deflection error averages samples in `averaging_interval=(0.5, 1.0)`.
Short runs without samples in that interval leave it missing, rather than
comparing a transient displacement with the steady analytical value.

Before advancing time, the runner measures the fluid momentum RHS, normalized by
gravity, over the whole fluid, a complete-support interior, and the remaining boundary
region. A separate check differentiates the affine field `f(x,y)=y` with the local
kernel gradient. This distinguishes failure of the local correction moments from
failure of the assembled conservative pressure force to balance gravity.

At each scheduled density reset, an independent raw-kernel sum predicts the Shepard
result using the **current evolved densities** before any update. A second sum freezes
fluid densities at their initial values while retaining current boundary volumes.
Agreement with the first, and measurable separation from the second, tests the actual
volume source rather than merely observing that the callback's timestamp advanced.
WCSPH pressure is checked immediately after the reset; EDAC pressure must be bitwise
unchanged by the density-only operation and continue evolving between resets.

Current output is marked `validation_version=2`. Earlier results used a different
WCSPH density model and an untranslated/constant-density pressure reference and must
be regenerated. The plotter rejects the old reference version and uses the summary
CSV as a campaign manifest, ignoring stale case folders from previous runs.

### Archived measured behavior and current regression bounds

The measurements below precede reference initialization and the latest WCSPH
fluid-side boundary flux. Their checked-in parameter files identify the archived
protocols; they are observations rather than regeneration targets for the new defaults.
Current post-fix outcomes are documented in [INITIAL_STATE.md](INITIAL_STATE.md) and
[POST_STARTUP_STABILITY.md](POST_STARTUP_STABILITY.md).

The complete 24-run summation matrix at `t=0.02` produced these final relative L2
errors (percent), excluding the redundant declaration-order repeats:

| Scheme | Plate resolution | None: density / pressure | Shepard: density / pressure | Mixed: density / pressure | Combined: density / pressure |
|:-------|-----------------:|-------------------------:|----------------------------:|--------------------------:|-----------------------------:|
| WCSPH | 3 | `2.427 / 99.35` | `1.296 / 98.15` | `2.433 / 99.28` | `1.498 / 98.70` |
| WCSPH | 5 | `1.763 / 101.7` | `0.949 / 101.1` | `1.762 / 101.3` | `1.108 / 100.8` |
| EDAC | 3 | `2.650 / 3.141` | `2.131 / 3.128` | `2.650 / 3.087` | `2.131 / 3.086` |
| EDAC | 5 | `1.937 / 3.784` | `1.586 / 3.744` | `1.936 / 3.637` | `1.585 / 3.632` |

The WCSPH startup pressure errors are `139.5–144.6%`, illustrating the density/EOS
consistency limitation above. Callback audits in all four continuity controls agree
with evolved-volume reconstruction within `1.1e-16`; frozen fluid volumes differ by
`1.31e-4–1.96e-4`. WCSPH post-reset EOS residual and EDAC reset-induced pressure change
are zero. Interface action/reaction residuals stay below `2.6e-15`, and the largest
normalized declaration-order difference is `3.4e-11`. Full-precision histories and
parameters are written per run; the checked-in reference CSVs retain eight significant digits.

An independent resolution-3 EDAC control holds density constant and starts from an
affine hydrostatic pressure field. Before any motion, it gives:

| Initial-state diagnostic | Uncorrected | Mixed, old two-endpoint boundary flux | Mixed, fluid-side Adami flux |
|:-------------------------|------------:|--------------------------------------:|-----------------------------:|
| Local affine-gradient residual | `0.0581` | `< 1e-12` | `< 1e-12` |
| Complete-support momentum residual / g | `0.00776` | `< 1e-10` | `< 1e-10` |
| Whole-fluid momentum residual / g | `0.0283` | `0.1148` | `0.0627` |

The mixed **local gradient** is consistent. Previously, the EDAC boundary pressure
force also used a separate gradient reconstructed on the truncated dummy-particle
support. Adami pressure is reconstructed from the fluid, so it must instead close
the fluid-side inter-particle pressure flux. Removing the second dummy gradient
reduces the boundary defect while preserving the exact structural reaction.
Conservative fluid-fluid forces still retain a free-surface consistency limitation;
the fix does not enforce exact hydrostatic balance on a truncated support.

Earlier matched runs through `t = 0.35` gave the following results after the boundary fix
(deflection averaged over the available samples in `[0.25, 0.35]`):

| Scheme | Plate resolution | Case | Final pressure error | Averaged deflection error | Initial whole-fluid momentum residual / g |
|:-------|-----------------:|:-----|---------------------:|--------------------------:|-----------------------------------------:|
| WCSPH | 3 | `none` | `1.957%` | `2.829%` | `0.0997` |
| WCSPH | 3 | `mixed` | `0.489%` | `3.023%` | `0.0555` |
| EDAC | 3 | `none` | `1.181%` | `2.297%` | `0.0708` |
| EDAC | 3 | `mixed` | `0.662%` | `2.385%` | `0.1084` |
| EDAC | 5 | `none` | `1.438%` | `15.684%` | `0.0936` |
| EDAC | 5 | `mixed` | `1.016%` | `15.671%` | `0.0854` |

At resolution 3, mixed-correction EDAC pressure error falls from `3.151%` before
the boundary fix to `0.662%` afterwards. It is now below the matched uncorrected
error at both checked resolutions: approximately 44% lower at resolution 3 and
29% lower at resolution 5. This is a matched improvement, not a claim of monotone
full-solver convergence; the coarse structural deflection errors remain visible.
`parameters.json` records `edac_adami_pressure_gradient="fluid-sided"` for these runs.

#### Complete 1.0-second study

The full 46-run study uses the ten matrix methods, two continuity controls and four
order repeats, all reaching `1.0 s`. The following ranges cover the ten forward matrix
methods at each scheme/resolution, with errors averaged over `[0.5, 1.0] s`:

| Scheme | Resolution | Mean pressure-error range | Mean plate-deflection-error range |
|:-------|-----------:|--------------------------:|----------------------------------:|
| WCSPH | 3 | `0.546–2.261%` | `2.679–2.838%` |
| EDAC | 3 | `0.756–2.288%` | `2.325–2.371%` |
| WCSPH | 5 | `0.824–2.026%` | `14.989–15.057%` |
| EDAC | 5 | `1.023–3.530%` | `15.313–15.372%` |

The longer horizon exposes deterioration hidden by the earlier short comparison.
EDAC mixed/no-Shepard at resolution 3 reaches a **42.27% sampled pressure-error peak
at `t=0.955 s`**, versus a maximum of 4.015% through `0.35 s`. An independent isolated-case
rerun reproduced the same summary. Resolution 5 mixed/no-Shepard has a 3.530% late
mean pressure error, exceeding its 2.231% uncorrected counterpart. Gradient-only and
mixed corrections therefore cannot be treated as interchangeable long-time accuracy claims.

The EDAC resolution-5 combined forward/reversed comparison also **exceeds the unchanged
`1e-6` order bound**: normalized density, pressure and deflection differences are
`2.402e-6`, `6.146e-5` and `1.438e-5`. The other three order pairs remain below the
bound. Panel H retains this exceedance, and `validation_reference_long_order.csv`
explicitly records `passed=false`; reproducing a measurement is not an accuracy pass.
The cause of this long-time order sensitivity remains unresolved.

Late deflection still oscillates: peak-to-peak spans across the matrix are approximately
6.4–12.7% of the analytical deflection, with half-window mean drift below 0.55%.
The study reports these quantities instead of assuming that reaching `1.0 s` establishes
a stationary solution. Structural under-resolution remains visible in the resolution-5
mean deflection. Full-precision histories and metadata are embedded in the report;
the versioned CSV/JSON files preserve compact reproducible summaries and known limitations.

#### Available four-level results

All ten forward WCSPH correction combinations have completed at every h-halving
level. Their late-window ranges are:

| Resolution | Mean pressure-error range | Mean plate-deflection-error range |
|-----------:|--------------------------:|----------------------------------:|
| 3 | `0.546–2.261%` | `2.679–2.838%` |
| 5 | `0.824–2.026%` | `14.989–15.057%` |
| 9 | `0.885–1.803%` | `12.895–12.923%` |
| 17 | `0.986–2.079%` | `7.435–7.488%` |

The reversal control at resolution 9 exceeds the unchanged `1e-6` criterion in pressure
and deflection (`1.158e-4` and `5.092e-4`), while its density difference is `8.914e-7`.
The close resolution-3 deflection remains a misleading coarse-grid coincidence.
EDAC fine-grid failures are recorded explicitly in the terminal case ledger;
completed-method envelopes are not presented as successful completion
of its entire matrix.

An instrumented fixed-grid rerun of EDAC mixed at resolution 9 records severe growth
before the grid cutoff: pressure error is 84.6% at the accepted sample `t=0.66 s`.
At the subsequent exception near `t=0.662713 s`, its saved previous state has maximum
speed `130.716 m/s`, minimum pressure `-1.550 MPa`, and 15 particles outside the nominal
side walls, with finite coordinates. The artificial sound speed is `50 m/s`.
The accepted-state deterioration is therefore already present before the grid error.
Streamed histories and state dumps are diagnostic evidence, kept under
`diagnostics/` with explicit variant metadata; the original failure remains in the
matched campaign ledger. The completed unbounded-cell-list control also fails:

| Diagnostic rerun, resolution 9 | Stop time [s] | Previous-state maximum speed [m/s] | Outcome |
|:------------------------------|--------------:|----------------------------------:|:--------|
| Mixed, original full grid | `0.662713` | `130.716` | Grid bounds exceeded |
| Mixed, unbounded dictionary grid | `0.643143` | `427.862` | Step size collapsed to `1.11e-16` |
| Continuity + reinit, original full grid | `0.832029` | `93.487` | Grid bounds exceeded |

Removing the grid-size restriction therefore does not establish a stable trajectory.
The unbounded control reaches positions well outside the tank and remains far from
the hydrostatic reference before its step-size abort. These are numerical robustness
findings for the tested configuration, not successful refinement measurements.

CI requires initial hydrostatic pressure error below `1e-4`, mixed affine-gradient
error below `1e-12`, and improvement of the complete-support momentum residual.
The short coupled runs additionally check density error below `0.04`, pressure error
below `0.1`, equation-of-state residual below `1e-12`, and
interface residual below `2e-12` with a nonzero interface load. Reversed histories
are compared with `rtol=1e-6` and `atol=1e-10`; the small tolerance accommodates
adaptive integration roundoff. CI also verifies that density reinitialization
actually ran after the initial time and that clamped particles carry a nonzero
part of the interface reaction.
An additional resolution-3 EDAC `none/mixed` regression runs through `t = 0.35`
and requires corrected final pressure error below 1%, at least 25% lower than
the matched baseline, no more than a 5% increase in peak pressure error, and averaged
analytical plate-deflection error below 5%. Pairwise and coupled tests additionally
verify the fluid-side wall closure's independence of the dummy gradient cache and
exact interface action/reaction for rigid bodies and TLSPH structures.

The summation-matrix CI regression uses a `0.002` time span and pressure bounds of
`1.5` for WCSPH and `0.1` for EDAC, based on the measured startup maxima at both
resolutions. These are regression/stability bounds for the stress cases, not physical
accuracy targets. It checks reference-initialized WCSPH against analytical initial
density/pressure bounds and uses compatible archived initial-state fixtures for EDAC
and continuity controls. It covers all four
summation configurations, both order repeats, four actual density resets, volume
residual below `1e-12`, frozen-volume separation above `1e-9`, immediate WCSPH EOS
consistency, and unchanged EDAC pressure during reinitialization.

### Isolating the resolution-5 deflection error

The approximately 15.7% under-deflection at resolution 5 is also present in the
original checked-in WCSPH/EDAC reference histories. It is reproduced by the **isolated
structural discretization**, with nominal hydrostatic traction and material gravity,
without pressure evolution or a time integrator:

```julia
trixi_include(@__MODULE__,
              joinpath(validation_dir(), "hydrostatic_water_column_2d",
                       "validation_structure.jl"))
```

The control assembles the sparse Hessian of the stress-free TLSPH elastic energy,
including strain contributions from clamped particles. It applies `ρ_fluid * g * H`
to the top row with half endpoint weights, plus gravity using the actual material
masses, and solves the linearized equilibrium. A central finite-difference check
against the production `kick!` verifies the stiffness action. The geometry, masses,
material constants, kernel, smoothing length and clamp indices are separately checked
against the FSI example at resolutions 3 and 5.

| Particles through thickness | Static central deflection / exact beam reference |
|---------------------------:|---------------------------------------------------------------:|
| 3 | `1.02273` |
| 5 | `0.84534` |
| 11 | `0.89528` |
| 21 | `0.93734` |
| 41 | `0.97759` |

Here the exact beam coefficient is `1/384`; the example's rounded `0.0026`
coefficient changes the reference by only 0.16%. The CSV records both references
and reports signed bias against the example's reference. Refinement reduces the
error beyond resolution 5; the close resolution-3 result is not evidence that the
structure is resolved. Shell particles have full `ρ Δx²` material masses, so their
represented volume exceeds the nominal plate volume at coarse resolutions. The
one-sided corrected gradients and clamped support also enter the discrete stiffness;
simply halving surface masses makes the resolution-5 error worse (about 69%
over-deflection). This is not a validated configuration fix.

Results and experiment settings are written to
`out_hydrostatic_structure/{hydrostatic_structure.csv,parameters.json}`. The reduced
regression checks RHS linearization, equilibrium residuals, example agreement, and
improvement from resolution 5 to 21. Continuum plate accuracy requires a structural
refinement check; changing fluid pressure correction alone cannot remove this error.
This control isolates the structural contribution rather than replacing the coupled
benchmark or adjusting its material parameters.

### PR-thread coverage

The shared benchmark implements the
[requested extension](https://github.com/trixi-framework/TrixiParticles.jl/pull/1286#issuecomment-6057087241):

| Request | Implementation / evidence |
|:--------|:--------------------------|
| Four matched summation-density correction combinations | `validation_summation_density.jl`, both schemes at resolutions 3 and 5 |
| Fluid/structure declaration-order lifecycle | Replayed time grids, density/pressure/deflection histories, `hydrostatic_correction_order.csv` |
| Evolved-volume reinitialization | Independent pre-update reconstruction and frozen-fluid-volume counterfactual; EDAC and WCSPH controls |
| Updated WCSPH EOS / independent EDAC pressure | Immediate reset audit plus history sensors and pressure-preservation regression |
| Pressure action/reaction and plate accuracy | Clamped-inclusive interface forces, nonzero-load check, original analytical late-time comparison |
| Reproducible multiresolution output and CI | JSON/CSV/parameter output, versioned reference summaries, reduced regressions |
| Release notes | Version 0.6.0 entry in `NEWS.md` |

### Single-file visual report

With `Plots` available:

```julia
trixi_include(@__MODULE__,
              joinpath(validation_dir(), "hydrostatic_water_column_2d",
                       "plot_correction_results.jl");
              output_directory="out_hydrostatic_corrections/short_run",
              operator_file="out/correction_convergence.csv",
              report_file="out/correction_validation.html")
```

Open `out/correction_validation.html` in a browser. The default expanded report contains
four four-panel figures; compact mode (`expanded=false`) contains two. Vector
figures, parameters, and downloadable source data are embedded in this single offline file.
Use `operator_file=nothing` for a coupled-only report; by default, the CSV under
`out` is included if it exists. Multiple coupled input directories can be supplied
with `hydrostatic_directories=("path/to/study1", "path/to/study2")`.
If the isolated-plate CSV exists at its default path, its signed refinement biases
are included in Figure 2's caption and its full CSV/settings are embedded. Override
`structure_file="path/to/hydrostatic_structure.csv"` to include a different run.

The coupled figure contains final/maximum density errors, final/maximum pressure
errors, normalized plate-deflection histories, and implementation checks. Run groups
identify scheme, resolution, source and final time; the individual ticks identify
corrections and reversed ordering. Each error panel has its own legend and percentage
ticks. Symbols show final errors and caps show maxima; numbers highlight the longest
uncorrected/mixed comparisons.

The analytical deflection becomes a unit reference in the history plot, and its
available averaging window is shaded. The legend reports signed mean errors for up
to eight long histories; short controls are retained in gray. The residual panel
combines clamped-inclusive interface force balance, WCSPH EOS consistency, and the
maximum declaration-order difference across mean-density, mean-pressure, and
deflection histories. Each is divided by its own regression bound, so all share a
limit of one; exact zeros have a separate bottom row. Full histories and source
settings remain available in the folded data section. See the
[correction report documentation](../corrections/README.md#single-file-visual-report)
for the theoretical guides and full report contents.

For the complete long campaign, panels E/F place correction methods on the rows
and scheme/base-density/resolution on the columns. Each row spells out the gradient
choice and Shepard on/off; the continuity-density control is named explicitly.
Cells show **late-window mean error** followed by the full-history maximum, so a
single favorable final sample cannot hide late oscillations. The caption states the
mean window, the blend factor, the two meanings of Shepard, and the reset interval.
Panel G shows the raw and mixed histories plus the min–max envelope over all forward
methods in each scheme/resolution. All individual histories, summaries, method
definitions and settings are embedded. Compact mode has two figures/eight panels;
the expanded report adds the figures described below.
With more than four configurations, scheme/base-density headers group the columns;
cell labels use two significant digits while downloads retain full precision. With
three or more matched resolutions, G shows deflection error against actual spacing,
including method min–max bands and the isolated structural control when available.
`hydrostatic_refinement.csv` records adjacent-level error ratios and observed orders
for density, pressure and deflection. Different protocols are kept separate; negative
orders retain deterioration, and no theoretical full-solver order is prescribed.
The CSV compares adjacent **available** levels and records the actual spacing ratio;
failed/missing levels are never interpolated or silently treated as h-halving steps.

The expanded report contains **four 2×2 figures, sixteen panels and four PDF pages**:
operator consistency (A–D), continuity-density hydrostatics (E–H), summation-density
hydrostatics (I–L), and cross-density initialization/transient/outcome checks (M–P).
Each hydrostatic figure covers WCSPH and EDAC at the four levels. Initialization
records are saved before time integration, so startup defects remain available even
when a trajectory fails early. Panel P covers every planned forward and reversed
case. Initial values below `1e-8%` use a labeled roundoff shelf in M/N; complete values
remain in the embedded data. Module-level `build_report(...; expanded=false)` retains
the compact layout for legacy or reduced reports.

# SPH correction validation

Two complementary studies validate the correction operators and their use in coupled
fluid–structure simulations.
The [current status and limitations](STATUS.md) distinguish local consistency checks
from stability and accuracy of the coupled solver.

## Manufactured operator study

- `convergence.jl`: reusable `CorrectionConvergence` module, including the manufactured
  fields, discrete operators, error norms, and refinement study.
- `validation_corrections.jl`: configurable validation runner that prints a Markdown
  table and saves the complete CSV data and Markdown report.
- `plot_correction_results.jl`: single-file visual report from the saved CSV and coupled histories.
- `plot_util.jl`: shared plotting, theoretical expectations, and self-contained HTML generation.

Run the default study from the repository root:

```bash
julia --project=. validation/corrections/validation_corrections.jl
```

The default resolutions are `N = (12, 24, 48, 96)` particles per coordinate direction.
Results are written to `out/correction_convergence.csv` and
`out/correction_convergence.md`. The existing direct entry point,
`julia --project=. validation/corrections/convergence.jl`, also prints the table and
writes the CSV.

Parameters can be changed with `trixi_include`, as in the other validation cases:

```julia
using TrixiParticles
trixi_include(@__MODULE__,
              joinpath(validation_dir(), "corrections", "validation_corrections.jl");
              resolutions=(12, 24, 48), output_directory="out/corrections_manual")
```

### Setup and interpretation

The particle patches fill the unit square and use a Wendland C6 kernel with
`h / Δx = 2`. Cubic manufactured scalar and pressure fields provide analytical values
and gradients. The pressure field vanishes at the left free surface.

The study records:

- raw and Shepard-normalized interpolation;
- direct and difference gradients with no correction, kernel, gradient, blended
  gradient (`λ = 0.5`), and mixed kernel/gradient corrections;
- production summation density with and without Shepard correction;
- continuity-density reinitialization from prescribed evolved density;
- every supported correction/pressure-formulation pairing, including combined Shepard
  density and mixed-gradient correction, and constant-pressure responses.

The boundary sample is the leftmost particle row, away from corners. The interior
sample has complete symmetric kernel support. Pressure samples keep **both** pair
neighborhoods away from unrelated boundaries and therefore start at `N = 24`.

`l2_error` is the relative discrete L2 error against the analytical field. The
constant-pressure rows instead report the absolute RMS acceleration, since the
analytical acceleration is zero. `observed_scaling` is
`log(error_coarse / error_fine) / log(Δx_coarse / Δx_fine)`; it is undefined (`NaN` in
CSV and `-` in Markdown) for the first resolution of each series.

These are **local operator scalings**, with fixed neighbor count, rather than
convergence orders of the full SPH discretization. In particular, one-pass Shepard
summation density retains a boundary error, and conservative corrected pressure
forces do not generally converge on a truncated support at fixed `h / Δx`. The
fluid-system documentation contains the moment identities and measured scaling tables.

### Single-file visual report

Use an environment containing `TrixiParticles`, `CSV`, `DataFrames`, `JSON`, and `Plots`:

```julia
trixi_include(@__MODULE__,
              joinpath(validation_dir(), "corrections", "plot_correction_results.jl");
              output_directory="out/corrections_manual",
              hydrostatic_directories=("out_hydrostatic_corrections",),
              report_file="out/correction_validation.html")
```

Open **`out/correction_validation.html`** in a browser. The complete report is
**two 2×2 multipanel figures (eight panels)** in compact mode (`expanded=false`):

| Figure | Panel | Contents |
|:-------|:------|:---------|
| Operator consistency | A | Interpolation, production summation density, and density reinitialization |
| | B | Direct and difference gradients |
| | C | All conservative pressure-acceleration pairings |
| | D | All constant-pressure responses |
| Coupled benchmark | E | Final and maximum density errors for every run |
| | F | Final and maximum pressure errors for every run |
| | G | Normalized plate-deflection histories, analytical reference, and signed mean errors |
| | H | Force, EOS, ordering, and reinitialization residuals relative to their own bounds |

Dark, high-contrast correction colors and distinct symbols are shared by panels
B–G. Panel A gives each quantity/correction its own color, and panel H assigns colors
to diagnostics. In the operator figure, solid curves indicate the one-sided boundary
and dashed curves the symmetric interior; closed interior symbols are hollow.
For gradients and pressure operators, each curve shows the maximum
error over the supported variants; a shaded min–max band includes **every** variant.
This keeps all measurements represented without a separate plot for every pairing.
Theoretical `O(h)`, `O(h²)`, `O(h⁻¹)`, and plateau guides appear directly in the panels.
Every operator panel has its own boundary/interior legend key. Slope labels are kept
away from data, near-coincident maxima are identified, and the constant-pressure
panel explicitly reports its worst interior response relative to the null-response bound.

Density/pressure panels have individual legends, readable percentage ticks, and
run groups labeled with scheme, resolution, source, and final time. Correction-specific
symbols mark final errors; horizontal caps mark history maxima. Numeric labels show
the longest-duration uncorrected/mixed comparisons.
For a complete hydrostatic matrix, E/F instead use method rows and configuration
columns. Method names spell out gradient correction and Shepard on/off; columns name
scheme, base density model, resolution and duration. The full matrix crosses Shepard
off/on with gradient none/kernel/gradient/blend (`λ=0.5`)/mixed, plus the explicit
continuity control. When every run reaches its averaging window, cells show its mean
error and the full-history maximum; otherwise they retain final/maximum errors.

Deflection histories are normalized by the recorded analytical plate deflection when
all references are available, making the steady target one. Line patterns distinguish
scheme/resolution and staggered symbols identify corrections. With up to eight histories
covering an averaging window, the legend gives each run's signed mean deflection error;
short controls remain visible in gray. No histories are smoothed or discarded.
With more than eight late histories, G shows uncorrected and mixed curves plus the
min–max envelope of all forward methods for each configuration. Individual histories
remain embedded and all methods enter E/F/H. The figure caption defines each encoding,
the averaging interval, Shepard filtering versus scheduled resets, and the reset interval.

The residual panel divides each diagnostic by its own regression bound: force `2e-12`,
EOS `1e-12`, and declaration order `1e-6`. The common acceptance limit is one, so a
residual is never compared with another diagnostic's tolerance. Exact zeros have a
separate bottom row below the logarithmic positive values. Force balance includes clamped
reactions; order is the largest normalized mean-density, mean-pressure, or deflection
history difference. Raw residuals remain in the embedded data. Blue circles, green
open diamonds, and purple stars identify the three diagnostics.

The file works offline: vector figures and downloadable full-precision CSV/JSON
data are embedded. Source settings and data links are folded below the figures.
Printing produces one landscape PDF page per figure: **two pages in compact mode**
or **four pages for the expanded sixteen-panel report**.
An optional `structure_file="path/to/hydrostatic_structure.csv"` adds the
[isolated TLSPH refinement control](../hydrostatic_water_column_2d/README.md#isolating-the-resolution-5-deflection-error)
to Figure 2's caption and embeds its CSV/settings without adding a panel or page.
The plotting entry points include `out_hydrostatic_structure/hydrostatic_structure.csv`
automatically when present. This separates structural under-resolution from the
fluid pressure-correction comparison.

Reference slopes are vertically offset from measured coarse values and are not error bounds.
Empirical regression bounds are explicitly labeled separately from theoretical targets.
An error floor is an allowed limitation of some operators, rather than a guaranteed
positive convergence order. The constant-pressure interior null-response target is
zero and has no prescribed truncation order. Full-solver convergence is not inferred
from local scaling.

The default report filename is `correction_validation.html` in `output_directory`.
The default coupled input is the complete `out_hydrostatic_long` study when present.
Otherwise, `out_hydrostatic_corrections` and `out_hydrostatic_summation_corrections`
are included when present. Use
`hydrostatic_directories=()` for an operator-only report. Multiple directories can
be supplied to include existing runs from different studies in the same file.
Hydrostatic histories must use the corrected `validation_version=2` reference;
earlier hydrostatic runs must be regenerated. The summary CSV identifies the
completed cases, so stale directories are excluded. Operator-only or coupled-only
input produces one four-panel figure. Use
`operators=(:interpolation, :difference_gradient)` only when a subset is desired.

## Coupled hydrostatic plate study

[`../hydrostatic_water_column_2d/validation_corrections.jl`](../hydrostatic_water_column_2d/validation_corrections.jl)
uses the existing elastic-plate benchmark to exercise time-dependent correction
updates, density reinitialization, and fluid–structure pressure reactions. Its
[README](../hydrostatic_water_column_2d/README.md) describes the cases, diagnostics,
output, and plotting workflow.

`validation/hydrostatic_water_column_2d/validation_summation_density.jl` supplies the
explicit four-way summation-density matrix requested in the PR thread: both schemes,
two resolutions, continuity/reinitialization controls, order repeats, and versioned
reference summaries. The continuity-density WCSPH setup supplies a physical baseline.
Current WCSPH summation cases use explicit reference-state initialization; the checked-in
short-study references preserve the earlier initialization protocol as labelled archives.
`validation/hydrostatic_water_column_2d/validation_long.jl` runs the full 2×5 physical
matrix in both schemes at the h-halving resolutions `(3, 5, 9, 17)` through `1.0 s`, with
statistics over `[0.5, 1.0] s`: 80 matrix runs, four continuity controls and eight order
repeats. `validation_refinement.jl` adds parallel workers, compatible-result reuse,
resumption, progress records, and automatic final report generation. The checked-in
long-study references currently contain the completed `(3, 5)` measurements.
That earlier 92-outcome campaign is a correction matrix within WCSPH continuity and
EDAC summation density, with one EDAC continuity control per level. The extended
`validation_refinement.jl` requests the full WCSPH/EDAC × continuity/summation
cross-product: 160 forward cases and 16 order repeats at the four levels. Compatible
completed trajectories and terminal failures are reused. The short, stabilized
WCSPH-summation study remains a different protocol.

For three or more matched resolutions, G becomes a full-solver plate-refinement plot
against `Δx`, with method ranges and an optional isolated-plate curve. Inter-level
density, pressure and deflection errors/orders are embedded as CSV. Protocols are
matched before comparing levels; negative observed orders remain visible. Compact mode
retains its eight-panel layout; expanded mode adds density-specific and cross-density
figures, with grouped scheme/density headers for wider matrices.
Failed/running cases remain visible as FAIL/RUN/WAIT cells and in the complete outcome
ledger. Panel G uses the intersection of completed methods across its displayed levels,
so changing method coverage cannot masquerade as convergence. A failed forward run
blocks its reversed-order control; unavailable comparisons are not counted as zeros.

## Automated validation

The expanded hydrostatic report has **four multipanel figures / sixteen panels**:
A–D operators, E–H continuity density, I–L summation density, and M–P startup,
pressure transients and all-case outcomes. It prints to four landscape PDF pages.
The compact module API remains available with `expanded=false`.

`test/validation/corrections.jl` checks reduced-resolution operator scaling, complete
envelope coverage, the four-panel layout, theoretical slope guides, and lossless embedded data.
`test/validation/hydrostatic_corrections.jl` runs short coupled
cases, checks the density/pressure histories and equation of state, compares reversed
system ordering, and verifies interface action/reaction including clamped particles.
Both run with the other validation cases in the `examples` CI job.
`test/validation/hydrostatic_matrix.jl` additionally exercises every method combination
in both schemes, checks the actual runtime corrections and reset schedules, and
verifies complete matrix coverage, explicit axis labels and late-statistic plotting.

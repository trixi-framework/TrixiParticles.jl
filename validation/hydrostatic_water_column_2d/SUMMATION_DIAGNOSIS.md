# WCSPH summation-density investigation

The current benchmark uses [reference-consistent initial density](INITIAL_STATE.md).
The startup measurements below describe the earlier uncorrected initial state.

The follow-up [timestep investigation](TIMESTEP_DIAGNOSIS.md) measures fluid-step caps,
structural substeps, tolerances and coupling variants after the predictor fix.

The 176-outcome study exposed both a startup consistency problem in this configuration
and a distinct implementation defect in Shepard/boundary staging. They require different
remedies; correcting RHS history dependence does not make the original setup accurate.

## 1. Interior kernel quadrature becomes a large pressure offset

For a uniform square lattice, Wendland C2 at `h/Δx = √2` gives

```text
σ = Δx² Σ W(r_b,h) = 1.0063469995574
ρ_sum = 1000 σ = 1006.3469995574 kg/m³
Cole(ρ_sum; ρ₀=1000, c=50, γ=7) = 16172.8483454 Pa
```

The complete two-metre hydrostatic load is approximately `19620 Pa`. In the hydrostatic
snapshot, the mean interior pressure bias is approximately `16623 Pa` at resolution 3
and `16621 Pa` at resolution 5, including the spatially varying compressed density.
Continuity-density controls give corresponding biases of only `-0.479 Pa` and `-0.239 Pa`.

This sum implements the declared density formula; the kernel's continuous normalization
is not the same as exact particle quadrature. Keeping `h/Δx` fixed preserves `σ`, so
refining the same lattice does not remove this interior offset. Increasing the ratio
reduces it (`σ=1.00120579634` at ratio 2, `1.00016242798` at ratio 3), but also changes
support requirements and is not by itself a complete free-surface remedy.

Reproduce the uniform-lattice measurement:

```julia
using TrixiParticles, LinearAlgebra
kernel = WendlandC2Kernel{2}()
h = sqrt(2.0) # Δx = 1; the dimensionless sum is spacing-independent
radius = ceil(Int, TrixiParticles.compact_support(kernel, h))
σ = sum(TrixiParticles.kernel(kernel, hypot(i, j), h)
        for i in -radius:radius, j in -radius:radius)
equation = StateEquationCole(; sound_speed=50.0, reference_density=1000.0, exponent=7)
(σ, equation(1000σ))
```

## 2. Free-surface truncation creates the opposite error

Missing support lowers density near the free surface. Even the one-pass Shepard
snapshot has mean density ratio about `0.9201` in the resolution-3 free-surface region
and pressures down to approximately `-263 kPa`. The normalizer uses `m_b/ρ_b` from
already-biased raw summation density, so it cannot recover an independently prescribed
physical density field or eliminate the uniform interior bias.

Large negative pressures cause tensile/sticking artifacts. Adami extrapolation can
also leave the real domain of the inverse Cole EOS: its nonlinear inverse requires
`p ≥ p_background - ρ₀c²/γ`, approximately `-357143 Pa` here. Several original failures
occur in this boundary-density inversion, rather than in the density summation loop.

Clipping alone is not an accuracy fix. A resolution-3 raw-summation control with fluid
and boundary pressure clipping, **without added viscosity**, reaches `1.0 s` but has
late mean pressure error `127.29%` and a deflection span of `3.819` times the nominal
reference magnitude. Its small mean-deflection error is therefore not evidence of a
resolved equilibrium. The corresponding Shepard control still leaves the grid.

## 3. Confirmed, corrected implementation defect

Previously, fluid Shepard coefficients read the EOS-backed boundary density left by
the previous RHS evaluation. With coordinates and velocities fixed, the resolution-3
hydrostatic probe changes pressure by `0.131258` of the reference pressure norm
between its first two RHS calls. Rejected trial evaluations can therefore contaminate
the next evaluation at a restored state.

The implementation now predicts EOS-backed Adami/Bernoulli boundary densities from
the **current raw WCSPH pressure**, before assembling any Shepard density normalizer.
After normalization, the usual final fluid-pressure and boundary-interpolation pass
uses the corrected density. The frozen-state probes at resolutions 3 and 5 then give
exactly zero density and pressure changes between repeated calls, including clipping
variants. Regression tests exercise wall, rigid-body and TLSPH boundaries, poisoned
caches, restored trial states, and reversed system declarations.

## Provenance and remaining limitation

The original 176-outcome report remains an archive of **pre-predictor** results. New
outputs record `density_boundary_policy="raw-pressure-predictor"`. Affected pre-fix
WCSPH summation/Shepard results and terminal failures are refused for post-fix reuse;
refinement comparisons keep the policies distinct. Earlier reference files are retained.
`validation_reference_summation_predictor.csv` contains the regenerated short stress
reference, with its own parameter JSON.

The remaining accuracy problem needs density-consistent discrete hydrostatic
initialization/quadrature and a defined free-surface pressure treatment. Resetting every
density to `ρ₀`, clipping pressures, or enlarging the search grid cannot establish those
properties. No full post-fix 176-case accuracy claim is made by these targeted tests.

Measured frozen-state region tables and clipping controls are stored under
`out/hydrostatic_corrections/summation_mechanism/`; they are diagnostics rather than
replacement results in the matched campaign.

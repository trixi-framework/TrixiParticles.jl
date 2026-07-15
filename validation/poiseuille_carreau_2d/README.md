# 2D Poiseuille Flow with Carreau-Yasuda Viscosity

This folder contains a periodic 2D Poiseuille validation case for the
`ViscosityCarreauYasuda` non-Newtonian viscosity model in TrixiParticles.jl.

The case checks whether the numerical solution approaches the steady
one-dimensional Carreau-Yasuda velocity profile in a pressure-driven channel.
It also includes `n = 1`, which recovers the Newtonian parabolic Poiseuille
profile. The setup is inspired by the Poiseuille validation in section 3.1 of
Coclite et al. (2020), but uses WCSPH with an equivalent body acceleration
instead of the MRT-LBM pressure-gradient implementation used there.

## Files

- `../../examples/fluid/poiseuille_carreau_2d.jl`: reusable example setup for a
  single Carreau-Yasuda power-law index.
- `validation_poiseuille_carreau_2d.jl`: validation runner. It runs one or more
  Carreau-Yasuda power-law indices, writes interpolated velocity profiles with
  `PostprocessCallback`, and checks the final relative L2 errors against
  validation bounds.
- `plot_carreau_comparison.jl`: helper script for manual inspection of the
  saved JSON profiles and error trends.

## Setup

- Channel height: `H = 1.0`
- Channel length: `L = 6H`
- Periodic direction: `x`
- Solid walls: lower and upper `y` boundaries
- Reference density: `rho0 = 1000.0`
- Zero-shear kinematic viscosity: `nu0 = 1.0e-3`
- Infinite-shear kinematic viscosity: `nu_inf = 0.0`
- Yasuda transition parameter: `a = 2.0`
- Default power-law indices: `n = (1.0, 1.5, 0.5, 0.25)`

## Analytical Profile

For each `n`, the script computes the steady profile by solving the implicit
Carreau-Yasuda stress relation

```text
tau(y) = rho0 * nu(gammadot) * gammadot
```

where `tau(y) = dpdx * abs(y - H / 2)`. The resulting shear-rate profile is
integrated from the wall to the centerline to obtain `u_x(y)`.

The validation records interpolated centerline velocity profiles with
`PostprocessCallback`. The error against the analytical profile is computed
after the run from those JSON files.

## Running

Default run:

```bash
julia --project=test validation/poiseuille_carreau_2d/validation_poiseuille_carreau_2d.jl
```

The default uses `ny = 50`, `t_end_factor = 0.1`, and analytical initial
conditions so it can be run quickly while still checking the numerical profile
against the analytical solution.

To change parameters from an interactive Julia session, use `trixi_include`:

```julia
using TrixiParticles
trixi_include(@__MODULE__,
              joinpath(validation_dir(), "poiseuille_carreau_2d",
                       "validation_poiseuille_carreau_2d.jl");
              ny=200, t_end_factor=0.02,
              n_values=(0.25, 0.5, 1.0, 1.5))
```

The initial condition mode is one of `:newtonian`, `:analytical`, or `:zero`.

Results are written to:

```text
out_poiseuille_carreau/n_<n>/
```

Each case contains VTU output and a
`validation_run_poiseuille_carreau_2d_*.json` profile history.

The validation runner checks the final relative L2 error with the following
bounds:

```text
n = 0.25: relative L2 <= 0.06
n = 0.5:  relative L2 <= 0.06
n = 1.0:  relative L2 <= 0.06
n = 1.5:  relative L2 <= 0.06
```

The relative L2 error is the main quantitative comparison metric. The maximum
absolute error is also written to the CSV/JSON files, but it should not be
compared directly across different `n` values without normalization because the
velocity scale changes strongly for shear-thinning cases such as `n = 0.25`.

## Plotting

After a run, create comparison plots with:

```bash
julia --project=test -e 'append!(ARGS, ["out_poiseuille_carreau"]); include("validation/poiseuille_carreau_2d/plot_carreau_comparison.jl")'
```

The plotting helper uses optional plotting packages that are available through
the test environment. They are not runtime dependencies of TrixiParticles.jl.

# 2D Poiseuille Flow with Carreau-Yasuda Viscosity

This folder contains a periodic 2D Poiseuille validation case for the
`ViscosityCarreauYasuda` non-Newtonian viscosity model in TrixiParticles.jl.

The default run starts from the steady one-dimensional Carreau-Yasuda velocity
profile and checks the numerical solution against that profile after time integration.
It also includes `n = 1`, which recovers the Newtonian parabolic Poiseuille
profile. The setup is inspired by the Poiseuille validation in section 3.1 of
[Coclite et al. (2020)](https://doi.org/10.3390/nano10112190), but uses WCSPH
with an equivalent body acceleration instead of the MRT-LBM pressure-gradient
implementation used there.

## Files

- `../../examples/fluid/poiseuille_carreau_2d.jl`: reusable example setup for a
  single Carreau-Yasuda power-law index.
- `validation_poiseuille_carreau_2d.jl`: validation runner. It runs one or more
  Carreau-Yasuda power-law indices, writes interpolated velocity profiles with
  `PostprocessCallback`, and checks the final relative L2 errors against
  validation bounds.
- `validation_util.jl`: shared velocity-error calculation for validation and plotting.
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

Here `dpdx` is the magnitude of the driving pressure gradient and
`tau(y) = dpdx * abs(y - H / 2)` is the shear-stress magnitude. The script solves
the stress relation by bisection and integrates the shear rate from the wall
toward the centerline using the trapezoidal rule to obtain `u_x(y)`.

The validation records interpolated velocity profiles across the channel at
`x = L / 2` with `PostprocessCallback`. The error against the analytical profile is computed
after the run from those JSON files.

## Running

Default run:

```julia
using TrixiParticles
include(joinpath(validation_dir(), "poiseuille_carreau_2d",
                 "validation_poiseuille_carreau_2d.jl"))
```

The default uses `ny = 50`, `t_end_factor = 0.1`, analytical initial conditions,
`WendlandC2Kernel`, no particle shifting, and a timestamped output directory.
Error bounds are checked by default for `initial_condition_mode=:analytical`.
For `:newtonian` and `:zero`, errors are still computed, but the bounds are not
checked unless `check_error_bounds=true` is passed.

To change parameters from an interactive Julia session, use `trixi_include`:

```julia
using TrixiParticles
trixi_include(@__MODULE__,
              joinpath(validation_dir(), "poiseuille_carreau_2d",
                       "validation_poiseuille_carreau_2d.jl");
              ny=200, t_end_factor=0.02,
              n_values=(0.25, 0.5, 1.0, 1.5),
              output_root="out_poiseuille_carreau/manual_run")
```

The initial condition mode is one of `:newtonian`, `:analytical`, or
`:zero`. The viscosity model is one of `:carreau` or `:newtonian`; the
plain Newtonian option uses the same periodic channel setup with
`ViscosityAdami(nu=nu0)`. Use `n_values=(1.0,)` with this option so the
initial profile and analytical comparison also use the Newtonian limit.

Results are written to:

```text
out_poiseuille_carreau/run_<timestamp>/n_<n>/
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

The final errors are available in `final_relative_l2_errors` and
`final_max_velocity_errors`. The JSON files contain velocity profile histories.
Wall velocities are included when interpolating the profiles used for both the
plots and the error calculation.

## Plotting

After running the validation in the same Julia session, display comparison plots
for that run with:

```julia
trixi_include(joinpath(validation_dir(), "poiseuille_carreau_2d",
                       "plot_carreau_comparison.jl");
              output_directory=output_root)
```

For an earlier run, set `output_directory` to its
`"out_poiseuille_carreau/run_<timestamp>"` directory. The plotting default is
`"out_poiseuille_carreau"`; timestamped runs require the subdirectory explicitly.
The plotting script requires `Glob` and `Plots` in the active Julia environment.
Figures are displayed without saving. Pass `save_figures=true` to also save PNG
files in the selected output directory and its case subdirectories.

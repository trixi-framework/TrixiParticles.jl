# Oscillating beam validation

The following files are provided:

- `validation_oscillating_beam_2d.jl` runs the validation simulation and compares
  the tip displacement with the corresponding TrixiParticles.jl reference result.
- `plot_oscillating_beam_results.jl` plots the tip displacement in both coordinate
  directions and compares it with the Turek and Hron (2006) reference data.
- `plot_oscillating_beam_results_oconnor.jl` plots the tip y-deflection for all
  available resolutions and compares it with the results of O'Connor and Rogers
  (2021) and Turek and Hron (2006).
- `validation_reference_{5,9,17,33,65}.json` contains TrixiParticles.jl reference
  results. The number in the filename is the number of particles across the beam
  thickness. Thus, for a beam thickness `t_s` and particle spacing `dp`, these
  files correspond to `t_s / dp = {4,8,16,32,64}`.
- `reference_oconnor.csv` contains reference results from O'Connor and Rogers
  (2021), *A fluid--structure interaction model for free-surface flows and
  flexible structures using smoothed particle hydrodynamics on a GPU*,
  [doi:10.1016/j.jfluidstructs.2021.103312](https://doi.org/10.1016/j.jfluidstructs.2021.103312).
- `reference_turek.csv` contains reference data extracted from
  `csm3_l4_t0p005.point`, available from the
  [FEATFLOW FSI benchmark](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/fsi_benchmark/fsi_tests/fsi_csm_tests.html)
  by Turek and Hron (2006).

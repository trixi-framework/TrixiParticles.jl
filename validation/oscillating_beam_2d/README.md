# Oscillating beam validation

The following files are provided:

- `validation_oscillating_beam_2d_oconnor.jl` runs the validation simulation and compares
  the tip displacement with the corresponding TrixiParticles.jl reference result.
  It uses the setup of O'Connor and Rogers (2021), which does not converge to the
  solution by Turek and Hron (2006): the particles are placed on the shell, so the
  discrete beam is one particle spacing thicker than the actual beam, and the elastic
  beam extends inside the cylinder, where it is only bonded to the clamped cylinder
  particles. The deflection is tracked at the particle in the middle of the tip of the
  beam and additionally with a `StructureMotionCalculator` at the center of the tip.
- `validation_oscillating_beam_2d_turek.jl` runs the validation simulation with the
  setup of `examples/structure/oscillating_beam_2d.jl`, which uses cell-centered
  particles, and the elastic beam is glued to the rigid cylinder, i.e., it starts at the
  cylinder surface and all particles inside the cylinder are clamped. This setup
  converges to the solution by Turek and Hron (2006). There is no particle at the tip of
  the beam, so the deflection is tracked with a `StructureMotionCalculator` at the center
  of the tip.
- `plot_oscillating_beam_results.jl` plots the tip displacement in both coordinate
  directions and compares it with the Turek and Hron (2006) reference data.
  Set `include_sim_results = true` to also plot the results in the `out` folder.
- `plot_oscillating_beam_convergence.jl` plots the maximum tip y-deflection of the
  first swing over the resolution for both setups and compares it with Turek and Hron
  (2006).
- `validation_reference_{oconnor,turek}_{5,9,17,33,65,129,257,513}.json` contains
  TrixiParticles.jl reference results. The number in the filename is the number of
  particles across the beam thickness. Thus, for a beam thickness `t_s` and particle
  spacing `dp`, these files correspond to `t_s / dp = {4,8,16,...,512}` for the O'Connor
  setup and `t_s / dp = {5,9,17,...,513}` for the Turek setup. The files for 129, 257,
  and 513 only contain the first swing of the beam (`tspan = (0, 0.6)`).
- `reference_turek.csv` contains reference data extracted from
  `csm3_l4_t0p005.point`, available from the
  [FEATFLOW FSI benchmark](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/fsi_benchmark/fsi_tests/fsi_csm_tests.html)
  by Turek and Hron (2006).

The files in this folder provide a 2D oscillating drop validation case for
TrixiParticles.jl based on the following reference:

M. Antuono, S. Marrone, A. Colagrossi, B. Bouscasse.
"Energy balance in the δ-SPH scheme".
In: Computer Methods in Applied Mechanics and Engineering, Volume 289 (2015),
pages 209–226.
https://doi.org/10.1016/j.cma.2015.02.004

The following files are provided here:

1. `validation_oscillating_drop_2d.jl`: Script that runs the oscillating drop example
   at coarse resolution `fluid_particle_spacing = 0.05` (`R / dx = 20`) and
   extracts the energy components plotted in Figure 3 of Antuono et al. (2015). The
   resolution used in the paper is `R / dx = 200`. To run the validation at this
   resolution, use
   `trixi_include("validation_oscillating_drop_2d.jl", fluid_particle_spacing=0.005)`.
2. `plot_oscillating_drop_energy.jl`: Script to plot the current simulation results
   from the `out` directory against the reference results from Antuono et al. (2015).
   The script also plots the provided reference results produced with TrixiParticles.jl.
   This allows for regression testing and for analyzing the behavior of the simulation
   when changing model or parameters.
3. `energy_quantities.jl`: Definitions of the energy quantities extracted during the
   simulation.

The reference data from Antuono et al. (2015) are provided in
`reference_antuono_2015.csv`.

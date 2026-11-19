The files in this folder provide a 2D vortex street validation case for
TrixiParticles.jl based on the following reference:

A. Tafuni, J. M. Domínguez, R. Vacondio, A. J. C. Crespo.
"A versatile algorithm for the treatment of open boundary conditions in smoothed
particle hydrodynamics GPU models".
In: Computer Methods in Applied Mechanics and Engineering, Volume 342 (2018),
pages 604–624.
https://doi.org/10.1016/j.cma.2018.08.004

The following files are provided here:

1. `validation_vortex_street_2d.jl`: Script that runs the vortex street example at
   `Re = 200` and records the drag and lift coefficients of the cylinder as well as
   the transverse velocity at a point in the wake of the cylinder.
   The resolution used in the paper is `d / dx = 100`. The default resolution in this
   script is `d / dx = 20`. To run the validation at a different resolution, use e.g.
   `trixi_include("validation_vortex_street_2d.jl", resolution_factor=0.02)`
   for `d / dx = 50`.
2. `plot_vortex_street_reference_2d.jl`: Script to plot the provided reference results
   produced with TrixiParticles.jl and to compute the Strouhal number and the force
   coefficients of the periodic vortex shedding.
   These values can be compared to the values reported in Tafuni et al. (2018).
   This allows for regression testing and for analyzing the behavior of the simulation
   when changing model or parameters.
   The resolution can be selected with `resolution_factor`.
3. `plot_vortex_street_result_2d.jl`: The same as `plot_vortex_street_reference_2d.jl`,
   but reading the current simulation results from the `out` directory.

The reference results produced with TrixiParticles.jl are provided in
`resulting_force_dp*.json`, where the number denotes the resolution `d / dx`.

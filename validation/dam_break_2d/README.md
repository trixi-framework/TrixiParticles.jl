The files in this folder provide a 2D dam break validation case for TrixiParticles.jl
based on the following references:
1. J. J. De Courcy, T. C. S. Rendall, L. Constantin, B. Titurus, J. E. Cooper.
   "Incompressible δ-SPH via artificial compressibility".
   In: Computer Methods in Applied Mechanics and Engineering, Volume 420 (2024),
   https://doi.org/10.1016/j.cma.2023.116700
2. S. Marrone, M. Antuono, A. Colagrossi, G. Colicchio, D. le Touzé, G. Graziani.
   "δ-SPH model for simulating violent impact flows".
   In: Computer Methods in Applied Mechanics and Engineering, Volume 200, Issues 13–16 (2011), pages 1526–1542.
   https://doi.org/10.1016/J.CMA.2010.12.016
3. J. C. Martin, W. J. Moyce, William George Penney, A. T. Price, C. K. Thornhill.
   "Part IV. An experimental study of the collapse of liquid columns on a rigid horizontal plane", Phil. Tran. R. Soc. London, Volume 244, Issue 882 (1952).
   https://doi.org/10.1098/rsta.1952.0006

The following files are provided here:

1. `setup_marrone_2011.jl`: Setup file for the dam break simulation based on Marrone et al. (2011).
   This can be used to reproduce the images of the breaking wave in Marrone et al. (2011).
2. `validation_dam_break_2d.jl`: Script to run the dam break simulation
   based on De Courcy et al. (2024) and extract pressure sensor data.
   **Note**: This paper used particle shifting, which is not yet implemented
   in TrixiParticles.jl for free surface simulations.
3. `plot_pressure_sensors.jl`: Script to plot the pressure sensor data and compare with
   De Courcy et al. (2024). This script can plot both the reference data included
   in this folder and simulation results from the `out` folder (if available).
4. `plot_surge_front.jl`: Script to plot the surge front position and compare with
   Martin et al. (1952). This script can plot both the reference data included
   in this folder and simulation results from the `out` folder (if available).

Note that the reference files for resolutions 40 and 80 were obtained with `SerialUpdate()`
and `--check-bounds=yes` on an x86 CPU.
The reference file for resolution 400 was obtained with `update_strategy=nothing`
and `--check-bounds=auto`, also on an x86 CPU.

# Oscillating drop energy validation

This validation extracts the energy components plotted for the 2D oscillating drop in
Antuono, Colagrossi, and Marrone (2015), Fig. 3.

Run the validation with

```bash
julia --project=run validation/oscillating_drop_2d/validation_oscillating_drop_2d.jl
```

The script writes `out/validation_result_oscillating_drop_2d_dx_*.json` and `.csv`.
The default run uses `fluid_particle_spacing = 0.05`, which is much coarser than the
paper's `R / dx = 200` setup. Increase the resolution in the validation script for a
paper-quality run.

Create the energy plot with

```bash
julia --project=run validation/oscillating_drop_2d/plot_oscillating_drop_energy.jl
```

The validation uses the linear Cole equation of state (`exponent = 1`) and no artificial
viscosity so the compressible energy and delta-SPH heat diagnostics match the balance
used in the paper. The base example file is left unchanged.

# Tutorials

Choose a tutorial based on the task in front of you.

> New to TrixiParticles.jl? Start with [Setting up your simulation from scratch](tutorials/tut_setup.md).

## Recommended Path

1. [Setting up your simulation from scratch](tutorials/tut_setup.md): learn the structure of a simulation file and run a complete WCSPH example.
2. [Modifying or extending components of TrixiParticles.jl within a simulation file](tutorials/tut_custom_kernel.md): replace selected parts of an existing setup without cloning the package.
3. [Setting up a 2D simulation from geometry files](tutorials/tut_2d_geometry.md): load 2D geometry files, turn them into filled wall regions, and combine them with standard 2D fluid blocks.
4. [Particle packing tutorial](tutorials/tut_packing.md): build a body-fitted particle configuration for complex geometries.

## Tutorials

### [Setting up your simulation from scratch](tutorials/tut_setup.md)

```@raw html
<img src="../tutorials/tut_setup_plot_tank.png"
     alt="Rectangular tank setup used in the first tutorial"
     style="max-width: 360px; width: 100%; border-radius: 12px;" />
```

Build a complete weakly compressible SPH dam break setup from particle spacing through semidiscretization, callbacks, and time integration.

- Focus: initial conditions, systems, semidiscretization, callbacks
- Choose this if: you want the full workflow from a minimal example

### [Modifying or extending components of TrixiParticles.jl within a simulation file](tutorials/tut_custom_kernel.md)

```@raw html
<img src="../tutorials/tut_custom_kernel_plot2.png"
     alt="Kernel comparison plot from the custom kernel tutorial"
     style="max-width: 360px; width: 100%; border-radius: 12px;" />
```

Start from an existing simulation and replace pieces such as the smoothing kernel directly in the file you run.

- Focus: `trixi_include`, custom kernels, rapid iteration
- Choose this if: you want to prototype changes without rewriting a full setup

### [Setting up a 2D simulation from geometry files](tutorials/tut_2d_geometry.md)

```@raw html
<img src="../tutorials/tut_2d_geometry_plot.png"
     alt="2D pipe and coastline geometries converted to wall and fluid particles"
     style="max-width: 360px; width: 100%; border-radius: 12px;" />
```

Load 2D geometry files, fill them with particles using `ComplexShape`, and build genuine 2D setups such as a curved pipe and a coastline dam break.

- Focus: `load_geometry`, `ComplexShape`, `setdiff`, 2D `Polygon`s
- Choose this if: you want a true 2D setup from line-based geometry data

### [Particle packing tutorial](tutorials/tut_packing.md)

```@raw html
<img src="https://github.com/user-attachments/assets/0f7aba29-3cf7-4ec1-8c95-841e72fe620d"
     alt="Packed particle configuration for a complex geometry"
     style="max-width: 360px; width: 100%; border-radius: 12px;" />
```

Go from a geometry file to a packed particle distribution using signed distance fields together with boundary and interior sampling.

- Focus: geometry import, signed distance fields, boundary sampling, `ParticlePackingSystem`
- Choose this if: you need body-fitted particles for complex shapes

See also [Getting started](getting_started.md) and [Examples](examples.md).

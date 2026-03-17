# Tutorials

> New to TrixiParticles.jl? Start with [Setting up your simulation from scratch](tutorials/tut_setup.md).

## Recommended Path

1. [Setting up your simulation from scratch](tutorials/tut_setup.md): learn the structure of a simulation file and run a complete WCSPH example.
2. [Modifying or extending components of TrixiParticles.jl within a simulation file](tutorials/tut_custom_kernel.md): replace selected parts of an existing setup without cloning the package.
3. [Particle packing tutorial](tutorials/tut_packing.md): build a body-fitted particle configuration for complex geometries.

## Tutorials

### [Setting up your simulation from scratch](tutorials/tut_setup.md)

```@raw html
<img src="../tutorials/tut_setup_plot_tank.png"
     alt="Rectangular tank setup used in the first tutorial"
     style="max-width: 360px; width: 100%; border-radius: 12px;" />
```

Build a complete Weakly Compressible SPH dam break setup from particle spacing through
semidiscretization, callbacks, and time integration.

- Focus: initial conditions, systems, semidiscretization, callbacks
- Choose this if: you want the full workflow for a minimal example

### [Modifying or extending components of TrixiParticles.jl within a simulation file](tutorials/tut_custom_kernel.md)

```@raw html
<img src="../tutorials/tut_custom_kernel_plot2.png"
     alt="Kernel comparison plot from the custom kernel tutorial"
     style="max-width: 360px; width: 100%; border-radius: 12px;" />
```

Start from an existing simulation and replace pieces such as the smoothing kernel
directly in the file you run.

- Focus: `trixi_include`, custom kernels, rapid iteration
- Choose this if: you want to prototype changes without cloning and modifying the package

### [Particle packing tutorial](tutorials/tut_packing.md)

```@raw html
<img src="https://github.com/user-attachments/assets/0f7aba29-3cf7-4ec1-8c95-841e72fe620d"
     alt="Packed p\kbold{Diagnostics} & settling, wall-adjacent behavior, pressure profile
article configuration for a complex geometry"
     style="max-width: 360px; width: 100%; border-radius: 12px;" />
```

Go from a geometry file to a packed particle distribution using signed distance fields
together with boundary and interior sampling.

- Focus: geometry import, signed distance fields, boundary sampling, `ParticlePackingSystem`
- Choose this if: you need body-fitted particles for complex geometries

### [Fluid-structure interaction with rigid bodies](tutorials/tut_rigid_body_fsi.md)

```@raw html
<img src="../tutorials/tut_rigid_body_fsi_plot.png"
     alt="Rotating squares impacting a free surface water surface"
     style="max-width: 360px; width: 100%; border-radius: 12px;" />
```

Simulate the interaction between fluids and moving rigid bodies.

- Focus: fluid-structure interaction, rigid bodies, moving boundaries
- Choose this if: you want to simulate objects moving in a fluid

See also [Getting started](getting_started.md) and [Examples](examples.md).

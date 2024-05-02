# Changelog

TrixiParticles.jl follows the interpretation of [semantic versioning (semver)](https://julialang.github.io/Pkg.jl/dev/compatibility/#Version-specifier-format-1)
used in the Julia ecosystem. Notable changes will be documented in this file for human readability. 
We aim at 3 to 4 month between major release versions and about 2 weeks between minor versions. 

## Version 0.2.x

### Highlights

### Added

### Removed

### Deprecated


## Version 0.1.2

### Highlights

#### Discrete Element Method
A basic implementation of the discrete element method was added.


## Version 0.1.1

### Added
A surface tension and adhesion model based on the work by Akinci et al., "Versatile Surface Tension and Adhesion for SPH Fluids", 2013 was added to WCSPH

# Pre Initial Release (v0.1.0)
This section summarizes the initial features that TrixiParticles.jl was released with.

## Highlights
### EDAC
An implementation of EDAC (Entropically Damped Artificial Compressibility) was added,
which allows for more stable simulations compared to basic WCSPH and reduces spurious pressure oscillations.

### WCSPH
An implementation of WCSPH (Weakly Compressible Smoothed Particle Hydrodynamics), which is the classical SPH approach.

Features:
- Correction schemes (Shepard (0. Order) ... MixedKernelGradient (1. Order))
- Density reinitialization
- Kernel summation and Continuity equation density formulations
- Flexible boundary conditions e.g. dummy particles with Adami pressure extrapolation, pressure zeroing, pressure mirroring...
- Moving boundaries
- Density diffusion based on the models by Molteni & Colagrossi (2009), Ferrari et al. (2009) and Antuono et al. (2010).


### TLSPH
An implementation of TLSPH (Total Lagrangian Smoothed Particle Hydrodynamics) for solid bodies enabling FSI (Fluid Structure Interactions).

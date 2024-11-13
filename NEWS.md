# Changelog

TrixiParticles.jl follows the interpretation of [semantic versioning (semver)](https://julialang.github.io/Pkg.jl/dev/compatibility/#Version-specifier-format-1)
used in the Julia ecosystem. Notable changes will be documented in this file for human readability.

## Version 0.2.4

### Features

- Support for surface tension was added to EDAC (#539)

### Documentation

- GPU Support Documentation was added (#660)

### Fixes

- JuliaFormatter was updated from 1.0.42 to 1.0.62 (#661)
- Diverse Doc fixes (#663, #659, #637, #658, #664)
- Simulations can be run with `Float32` (#662)

### Refactored

- Surface normal calculation was moved from surface_tension.jl to surface_normal_sph.jl (#539)

## Version 0.2.3

### Highlights

Transport Velocity Formulation (TVF) based on the work of Ramachandran et al. "Entropically damped artiﬁcial compressibility for SPH" (2019) was added.

## Version 0.2.2

### Highlights

Hotfix for threaded sampling of complex geometries.

## Version 0.2.1

### Highlights

Particle sampling of complex geometries from `.stl` and `.asc` files.

## Version 0.2.0

### Removed

Use of the internal neighborhood search has been removed and replaced with PointNeighbors.jl.

## Development Cycle 0.1

### Highlights

#### Discrete Element Method

A basic implementation of the discrete element method was added.

#### Surface Tension and Adhesion Model

A surface tension and adhesion model based on the work by Akinci et al., "Versatile Surface Tension and Adhesion for SPH Fluids" (2013) was added to WCSPH.

#### Support for Open Boundaries

Open boundaries using the method of characteristics based on the work of Lastiwka et al., "Permeable and non-reflecting boundary conditions in SPH" (2009) were added for WCSPH and EDAC.

## Pre Initial Release (v0.1.0)

This section summarizes the initial features that TrixiParticles.jl was released with.

### Highlights
#### EDAC

An implementation of EDAC (Entropically Damped Artificial Compressibility) was added,
which allows for more stable simulations compared to basic WCSPH and reduces spurious pressure oscillations.

#### WCSPH

An implementation of WCSPH (Weakly Compressible Smoothed Particle Hydrodynamics), which is the classical SPH approach.

Features:

- Correction schemes (Shepard (0. Order) ... MixedKernelGradient (1. Order))
- Density reinitialization
- Kernel summation and Continuity equation density formulations
- Flexible boundary conditions e.g. dummy particles with Adami pressure extrapolation, pressure zeroing, pressure mirroring...
- Moving boundaries
- Density diffusion based on the models by Molteni & Colagrossi (2009), Ferrari et al. (2009) and Antuono et al. (2010).


#### TLSPH

An implementation of TLSPH (Total Lagrangian Smoothed Particle Hydrodynamics) for solid bodies enabling FSI (Fluid Structure Interactions).

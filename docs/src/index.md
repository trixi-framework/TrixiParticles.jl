# TrixiParticles.jl

**TrixiParticles.jl** is a high-performance simulation framework for particle-based methods in complex multiphysics applications. It combines an accessible user interface with an extensible architecture for developing new methods, while also providing GPU-accelerated execution.

TrixiParticles.jl focuses on the following use cases:

- Development of new particle-based methods and models through an extensible architecture that is not tied to a single numerical method.
- Accurate, reliable, and efficient physics-based modeling of complex multiphysics problems through a flexible configuration system, high performance, and a broad set of validation and test cases.
- Accessible simulation setup for educational purposes, including student projects, coursework, and thesis work, supported by extensive documentation and readable configuration files.

Its main features include:

## Features
- Incompressible Navier-Stokes flows
  - Methods: Weakly Compressible Smoothed Particle Hydrodynamics (WCSPH), Entropically Damped Artificial Compressibility (EDAC), Implicit Incompressible Smoothed Particle Hydrodynamics (IISPH)
  - Models: Surface Tension, Open Boundaries
- Structural mechanics
  - Methods: Total Lagrangian SPH (TLSPH), Discrete Element Method (DEM)
- Fluid-Structure Interaction
- Particle sampling of complex geometries from `.stl`, `.asc`, and `.dxf` files
- Output formats:
  - VTK
- GPU support for NVIDIA, AMD, and Apple devices

## Examples
```@raw html
<table align="center" border="0">
  <tr>
    <td align="center">
      <img src="https://github.com/trixi-framework/TrixiParticles.jl/assets/10238714/683e9363-5705-49cc-9a5c-3b47d73ea4b8" style="width: 80% !important;"/><br><figcaption>2D Dam Break</figcaption>
    </td>
    <td align="center">
      <img src="https://github.com/trixi-framework/TrixiParticles.jl/assets/10238714/c10faddf-0400-47c9-b225-f5d286a8ecb8" style="width: 80% !important;"/><br><figcaption>Moving Wall</figcaption>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/trixi-framework/TrixiParticles.jl/assets/10238714/e05ace63-e330-441a-a391-eda3d2764074" style="width: 80% !important;"/><br><figcaption>Oscillating Beam</figcaption>
    </td>
    <td align="center">
      <img src="https://github.com/trixi-framework/TrixiParticles.jl/assets/10238714/ada0d554-e0ba-44ed-923d-2b77ef252258" style="width: 80% !important;"/><br><figcaption>Dam Break with Elastic Plate</figcaption>
    </td>
  </tr>
</table>
```

## Quick Start
1. [Installation](@ref installation)
2. [Getting started](@ref getting_started)

If you have questions about **TrixiParticles.jl**, join our community [on Slack](https://join.slack.com/t/trixi-framework/shared_invite/zt-sgkc6ppw-6OXJqZAD5SPjBYqLd8MU~g) or open an issue.

## Getting Started with Development
If you want to contribute or extend the code, start with:

1. [Installation](@ref installation)
2. [Development](@ref development)
3. [Contributing](@ref)

# TrixiParticles.jl

**TrixiParticles.jl** is a high-performance particle simulation framework designed to overcome challenges of particle-based numerical methods in multiphysics applications. Existing frameworks often lack user-friendliness, involve complex configuration, and are not easily extensible for development of new methods. In the future we also want to provide seamless scalability from CPU to Exascale-level computing with GPU support. **TrixiParticles.jl** addresses these limitations with an intuitive interface, straightforward configuration, and an extensible design, facilitating efficient simulation setup and execution.

TrixiParticles.jl focuses on the following use cases:

- Development of new particle-based methods and models. By providing an extensible architecture to incorporate additional particle methods easily and not focusing on a single model or numerical method.
- Accurate, reliable and efficient physics-based modelling of complex multiphysics problems by providing a flexible configuration system, tools, high performance and a wide range of validation and test cases.
- Easy setup of accessible simulations for educational purposes, including student projects, coursework, and thesis work through extensive documentation, community engagement and readable configuration files.

Its features include:

## Features
- Incompressible Navier-Stokes
  - Methods: Weakly Compressible Smoothed Particle Hydrodynamics (WCSPH), Entropically Damped Artificial Compressibility (EDAC)
  - Models: Surface Tension
- Solid-body mechanics
  - Methods: Total Lagrangian SPH (TLSPH)
- Fluid-Structure Interaction
- Output formats:
  - VTK

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

## Quickstart
1. [Installation](@ref installation)
2. [Getting started](@ref getting_started)

If you have any questions concerning **TrixiParticles.jl** you can join our community [on Slack](https://join.slack.com/t/trixi-framework/shared_invite/zt-sgkc6ppw-6OXJqZAD5SPjBYqLd8MU~g) or open an issue with your question.

## Start with development
To get started with development have a look at these pages:

1. [Installation](@ref installation)
2. [Development](@ref development)
3. [Contributing](@ref)

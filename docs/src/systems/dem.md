# [Discrete Element Method](@id dem)

The Discrete Element Method (DEM) is a computational technique widely used in physics, engineering,
and applied mathematics for simulating the mechanical behavior of granular materials, such as powders,
sand, soil, or rock, as well as other discontinua. Unlike continuum mechanics that treats materials as
continuous, DEM considers individual particles or elements and their interactions. This approach provides
detailed insights into the micro-mechanical behavior of materials, making it particularly valuable
in fields such as geomechanics, material science, and mechanical engineering.

## Fundamental Principles

The core idea behind DEM is the discretization of a material system into a finite set of distinct,
interacting mass elements (particles). These elements (particles) can vary in shape, size, and properties, and
they interact with each other and possibly with their boundaries through contact forces and potential fields.
The motion and behavior of each mass element are governed by Newton's laws of motion, accounting for the forces
and moments acting upon them.

## API

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "structure", "discrete_element_method", "system.jl")]
```

### Contact Models

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "structure", "discrete_element_method", "contact_models.jl")]
```

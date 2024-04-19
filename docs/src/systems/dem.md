# [Discrete Element Method](@id dem)
The Discrete Element Method (DEM) is a computational technique widely used in physics, engineering,
and applied mathematics for simulating the dynamics of systems of particles. This method allows
for the analysis of the mechanical behavior of granular materials, such as powders, sand, soil, or rock,
as well as other discontinua. Unlike continuum mechanics that treats materials as continuous, DEM
considers individual particles or elements and their interactions. This approach provides detailed insights
into the micro-mechanical behavior of materials, making it particularly valuable in fields such as
geomechanics, material science, and mechanical engineering.

## Fundamental Principles
The core idea behind DEM is the discretization of a material system into a finite set of distinct,
interacting elements or particles. These particles can vary in shape, size, and properties, and 
they interact with each other and possibly with their boundaries through contact forces and potential fields.
The motion and behavior of each particle are governed by Newton's laws of motion, accounting for the forces
and moments acting upon them.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "solid", "discrete_element_method", "system.jl")]
```

## References
- N. Bićanić. "Discrete element methods". 
  In: Encyclopedia of Computational Mechanics, 2004.
  [doi: 10.1002/0470091355.ecm006.pub2]( https://doi.org/10.1002/0470091355.ecm006.pub2)

- P. Cundall and O. Strack. "A discrete numerical model for granular assemblies". 
  In: Geotechnique, vol. 29, 1979.
  [doi: 10.1680/geot.1979.29.1.47](https://doi.org/10.1680/geot.1979.29.1.47)

- A. Renzo and F. Maio. "Comparison of contact-force models for the simulation of collisions in DEM-based granular flow codes"
  In: Chemical Engineering Science, 2004.
  [doi: 10.1016/j.ces.2003.09.037](https://doi.org/10.1016/j.ces.2003.09.037)


# [Rigid Body SPH](@id rigid_body_sph)

Rigid bodies in TrixiParticles.jl are represented by particles whose motion is evolved
with rigid-body translation and rotation. This allows fluid-structure interaction while
keeping the structure kinematics rigid.

## API

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "structure", "rigid_body_sph", "system.jl")]
```

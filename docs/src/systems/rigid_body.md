# [Rigid Bodies](@id rigid_body)

Rigid bodies in TrixiParticles.jl are represented by particles whose motion is evolved
with rigid-body translation and rotation. This allows fluid-structure interaction while
keeping the structure kinematics rigid.

## API

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "structure", "rigid_body", "system.jl")]
```

### Contact Models

Rigid contact is configured through the `contact_model` keyword of
[`RigidBodySystem`](@ref). This is separate from the `boundary_model` used for
fluid-structure interaction; see [Boundary Models](@ref boundary_models) for that part of
the rigid-body setup.

The previous name `RigidBoundaryContactModel` remains available as a compatibility alias.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "structure", "rigid_body", "boundary_contact_models.jl")]
```

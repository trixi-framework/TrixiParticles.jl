# [Rigid Bodies](@id rigid_body)

Rigid bodies in TrixiParticles.jl are represented by particles whose motion is evolved
with rigid-body translation and rotation. This allows fluid-structure interaction while
keeping the structure kinematics rigid.

## Contact Model

Rigid contact is configured per rigid body through the `contact_model` keyword of
[`RigidBodySystem`](@ref). In `collision-basic`, this enables rigid-wall collisions with
the basic linear spring-dashpot law.

```@docs
    RigidContactModel
```

`RigidContactModel` stores:
- `normal_stiffness`: normal spring stiffness
- `normal_damping`: normal damping coefficient
- `contact_distance`: interaction cutoff distance

If `contact_distance == 0`, the particle spacing of the corresponding
[`RigidBodySystem`](@ref) is used. The previous name `RigidBoundaryContactModel` remains
available as a compatibility alias.

## API

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "structure", "rigid_body", "system.jl")]
```

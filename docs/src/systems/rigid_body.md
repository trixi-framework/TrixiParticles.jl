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

Rigid contact is configured through the contact model. This is separate from the
boundary model used for fluid-structure interaction; see
[Boundary Models](@ref boundary_models) for that part of the rigid-body setup.

`RigidContactModel` is shared by rigid-wall and rigid-rigid contact. In this first
porting step it provides the active normal spring-dashpot law for both interaction
paths, and the rigid-body runtime now also reports the contact diagnostics
`contact_count` and `max_contact_penetration` through system data and VTK output.

The user-facing constructor remains the normal-only configuration with
`normal_stiffness`, optional `normal_damping`, and optional `contact_distance`.
Stateful friction and wall-specific contact extensions are added in later porting
steps so the existing rigid-contact examples keep their current behavior on this branch.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "structure", "rigid_body", "contact_models.jl")]
```

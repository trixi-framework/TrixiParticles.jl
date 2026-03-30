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

`RigidContactModel` is the shared runtime model for both rigid-wall and rigid-rigid
contact. In this first porting step, the shared normal spring-dashpot force and the
contact diagnostics are active for both paths. The richer tangential/friction fields and
the wall-specific controls are already part of the runtime model and documented on the
type, but their stateful and wall-specific behavior is added in later steps of the port.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "structure", "rigid_body", "contact_models.jl")]
```

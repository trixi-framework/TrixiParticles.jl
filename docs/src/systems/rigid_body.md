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

`RigidContactModel` defines the rigid-contact law shared by rigid-wall and rigid-rigid
interaction. The always-active parameters are `normal_stiffness`, `normal_damping`, and
`contact_distance`.

For rigid-wall contact, the current implementation also supports tangential friction with
the parameters `static_friction_coefficient`, `kinetic_friction_coefficient`,
`tangential_stiffness`, `tangential_damping`, `stick_velocity_tolerance`, and
`penetration_slop`. When the tangential spring history is active, this requires
[`UpdateCallback`](@ref) so the tangential displacement cache is updated between time
steps.

The current implementation uses the same model for rigid-wall and rigid-rigid contact, but
the active behavior is not identical yet:

- rigid-wall contact groups penetrating wall neighbors into a small number of contact
  manifolds per rigid particle and applies one normal-plus-tangential contact force per
  manifold,
- rigid-rigid contact evaluates direct pairwise normal contact forces between rigid
  particles,
- and rigid-rigid remains normal-only for now, even though the tangential-history key
  design is already shared with the wall path.

`contact_distance` defines when contact starts. If `contact_distance == 0`, the
particle spacing of the `RigidBodySystem` is used when the contact model is adapted to
the runtime system.

If no `contact_model` is specified for a rigid body, rigid-wall and rigid-rigid contact
for that system are disabled.

For output and postprocessing, rigid bodies also expose the diagnostics
`contact_count` and `max_contact_penetration`. They are available through rigid-body
system data and VTK output.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "structure", "rigid_body", "contact_models.jl")]
```

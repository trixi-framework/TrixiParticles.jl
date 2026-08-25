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

Rigid-wall and rigid-rigid contact also support tangential friction with
the parameters `static_friction_coefficient`, `kinetic_friction_coefficient`,
`tangential_stiffness`, `tangential_damping`, `stick_velocity_tolerance`, and
`penetration_slop`. When the tangential spring history is active, this requires
`UpdateCallback(interval=1)` so the tangential displacement cache is updated after every
accepted time step. Sparse and time-periodic update callbacks are not supported for contact
history.

A frictional setup with a tangential spring must install the update callback alongside the
other callbacks used by the simulation:

```julia
contact_model = RigidContactModel(; normal_stiffness=2.0e4,
                                  normal_damping=20.0,
                                  static_friction_coefficient=0.6,
                                  kinetic_friction_coefficient=0.4,
                                  tangential_stiffness=1.0e4,
                                  tangential_damping=5.0)
rigid_system = RigidBodySystem(initial_condition; contact_model)
update_callback = UpdateCallback(interval=1)
```

Penalty contact also requires a timestep short enough to resolve its spring and damping
scales. Add `StepsizeCallback(cfl=0.5)` to contact simulations, or impose an equivalent
`dtmax`. The automatic estimate includes every active normal and tangential contact scale.

### Force Law

Let ``\delta = d_c - r - \delta_0`` be the effective penetration after subtracting
`penetration_slop`, and let ``v_n`` be relative velocity along the outward contact normal.
For walls with geometry normals, ``r`` is the particle separation projected onto that normal;
otherwise it is the radial particle separation.
The non-attractive normal-force magnitude is

```math
F_n = \max(k_n \delta - c_n v_n, 0).
```

With the convention used here, approaching particles have ``v_n < 0``, so normal damping
increases the repulsive force during approach. No contact force is applied when
``\delta \le 0``.

For tangential displacement history ``\boldsymbol{\xi}`` and slip velocity
``\boldsymbol{v}_t``, the trial force is

```math
\boldsymbol{F}_t^{\mathrm{trial}} =
-k_t \boldsymbol{\xi} - c_t \boldsymbol{v}_t.
```

The contact sticks while
``\lVert\boldsymbol{F}_t^{\mathrm{trial}}\rVert \le \mu_s F_n``. Otherwise the model
uses kinetic friction of limiting magnitude ``\mu_k F_n`` opposite the current slip
velocity. `stick_velocity_tolerance` supplies a `tanh` regularization close to zero slip
speed. At exactly zero slip speed, the restoring direction of the trial force is retained.

After an accepted time step of length ``\Delta t``, history is advanced and projected back
onto the current contact plane:

```math
\boldsymbol{\xi} \leftarrow
\left(\boldsymbol{I} - \boldsymbol{n}\boldsymbol{n}^{T}\right)
\left(\boldsymbol{\xi} + \Delta t\,\boldsymbol{v}_t\right).
```

The stored extension is capped at the static Coulomb limit. Initialization uses
``\Delta t = 0`` so contacts are registered without adding displacement before the first
accepted step. Rejected steps and intermediate Runge-Kutta stages never advance history.

### Contact Pairs

The same contact model is used for both contact paths:

- rigid-wall contact groups penetrating wall neighbors into a small number of contact
  manifolds per rigid particle and applies one normal-plus-tangential contact force per
  manifold,
- rigid-rigid contact evaluates direct pairwise normal-plus-tangential contact forces between
  rigid particles.

For rigid-rigid contact, normal and tangential stiffness and damping are averaged between
the two models. Contact distance is the larger value, friction coefficients are the smaller
values, and the larger stick-velocity tolerance and penetration slop are used. These
symmetric rules ensure that the two ordered interaction passes produce equal-and-opposite
contact forces.

If either rigid body has zero friction coefficients, the minimum-coefficient rule makes the
pair frictionless. A tangential spring on only one body can contribute to a pair only when
both bodies have nonzero friction coefficients.

### Wall Manifolds

When the wall `InitialCondition` provides `normals`, rigid-wall contact uses the wall geometry
instead of the radial direction between particle centers. The stored normal is oriented
toward the contacting rigid particle, and penetration is measured from the particle
separation projected onto that normal. The Euclidean neighborhood-search radius is expanded
by one wall spacing so a tangential offset between the rigid and wall grids cannot hide a
valid projected contact. This makes flat-wall forces insensitive to tangential particle
alignment and wall resolution.

Normals attached to a wall follow its `PrescribedMotion`; translations leave them unchanged
and rotations rotate them with the wall. If normals are absent or have zero length, contact
falls back to the radial particle-pair direction for compatibility with arbitrary particle
walls.

Here, a contact manifold is a discrete approximation of one locally smooth contact
patch. A rigid particle touching a flat wall will usually produce one manifold,
while corners or edges can produce several.
Tangential history is associated with persistent contact IDs obtained by matching manifold
anchor positions and normals between accepted steps. Transient manifold array slots are not
used as physical contact identities.

Each accepted manifold stores a weighted wall-position anchor and contact normal. On the
next accepted step, matching is restricted to the same rigid particle and wall system. A
candidate must be within one `contact_distance` and its normal must be within 60 degrees of
the previous normal. Matching is one-to-one; unmatched manifolds receive monotonically
increasing IDs. RHS stages may read this mapping but only the accepted-step callback may
change it.

The number of cached rigid-wall manifolds per rigid particle is controlled by the
`RigidBodySystem(...; max_manifolds=8)` keyword argument. If more wall-contact
patches are detected than cached manifold slots are available, the implementation
falls back to the best-matching existing manifold for that particle.

`contact_distance` defines when contact starts. If `contact_distance == 0`, the
particle spacing of the `RigidBodySystem` is used when the contact model is adapted to
the runtime system.

If no `contact_model` is specified for a rigid body, rigid-wall and rigid-rigid contact
for that system are disabled.

### Lifecycle and Limitations

Positive friction coefficients require positive tangential stiffness or damping. Contact
friction currently uses CPU-managed dictionaries for history and wall descriptors and is
therefore not supported on GPU backends.
Fresh semidiscretizations and restarts clear tangential contact history; coordinates and
velocities are restored, but static-friction memory is reinitialized. Restarting a live
system with nonempty history emits a warning before discarding that state.

The contact contribution to automatic time-step selection includes normal and tangential
elastic and damping scales. For effective contact mass ``m``, the active scales are
``\sqrt{m/k_n}``, ``m/c_n``, ``\sqrt{m/k_t}``, and ``m/c_t``; the smallest is used before
the global CFL factor is applied. Rigid-wall contact uses the rigid particle mass. A
rigid-rigid pair uses the reduced mass formed from the lightest particle in each body.

For output and postprocessing, rigid bodies also expose the diagnostics
`contact_count` and `max_contact_penetration`. They are available through rigid-body
system data and VTK output.

Custom `source_terms` are interpreted as particle accelerations, multiplied by material
particle mass, and reduced to a resultant force and torque before being applied. They can
therefore drive translation and rotation without exciting non-rigid particle motion. The
uniform `acceleration` keyword remains a prescribed body acceleration and is not included in
the resultant-force diagnostics.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "structure", "rigid_body", "contact_models.jl")]
```

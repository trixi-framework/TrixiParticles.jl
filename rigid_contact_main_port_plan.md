# Plan: Port Rigid Contact Functionality From `collision` Into `main`

## Purpose

This document describes how to move the rigid-contact functionality currently developed on
the `collision` branch into `main` through a sequence of reviewable pull requests.

The plan is intentionally not "merge the branch." The branch mixes:

- core rigid-wall contact functionality,
- API renaming and compatibility cleanup,
- tests, examples, and docs,
- and some unrelated infrastructure changes.

The goal is to port the useful functionality onto `main` while preserving `main`'s existing
rigid-rigid behavior and keeping each PR narrow enough to review with confidence.

## Core Porting Principles

### 1. Preserve rigid-rigid support

The current `main` implementation still has a working rigid-rigid `interact!` path in
`src/schemes/structure/rigid_body/rhs.jl`.

The current `collision` branch, by contrast, contains:

```julia
# Collisions between rigid bodies are not yet implemented
function interact!(..., particle_system::RigidBodySystem,
                   neighbor_system::RigidBodySystem, semi)
    return dv
end
```

That regression must not be ported.

Anything that is conceptually shared between rigid-wall and rigid-rigid contact should be
designed and merged as shared functionality:

- runtime contact model fields as they become active,
- common normal/tangential force helpers,
- shared diagnostics,
- and shared tangential contact-history infrastructure.

Anything that is fundamentally wall-specific can remain wall-specific:

- wall-contact manifolds,
- contact-patch normalization,
- and the current resting-contact projection fallback.

### 2. No standalone docs/examples/coverage PR

Examples, docs, and tests should ship in the same PR that introduces the functionality they
exercise or explain.

### 3. No dormant API or unused changes

Every PR should introduce only code, runtime fields, docs, or examples that are exercised
by behavior landing in that same PR.

In particular:

- do not add `RigidContactModel` parameters "for later",
- do not add helper functions that no active code path uses yet,
- and do not update examples to mention options whose implementation is still deferred.

If a field, helper, or example knob is not yet live, it belongs in the later PR where it
first becomes live.

### 4. Re-implement on top of `main`

Do not try to merge the branch tip wholesale or cherry-pick it indiscriminately.

For each PR:

- start from `main` or the previous PR in the stack,
- manually port the specific functionality,
- and explicitly keep or restore `main` behavior where the branch diverged in the wrong
  direction.

### 5. Prefer behavior-preserving defaults

When adding new `RigidContactModel` parameters, default values should preserve current
`main` behavior unless the user explicitly opts into the advanced path.

### 6. Separate shared features from wall-only features

This is the most important architectural rule in the stack.

Shared features should not be hidden in wall-specific files or APIs, and wall-specific
features should not accidentally redefine the rigid-rigid contract.

## Current State Summary

### `main`

`main` currently provides:

- a minimal `RigidContactModel` with only `normal_stiffness`, `normal_damping`, and
  `contact_distance`,
- rigid-wall contact using manifold reduction but a normal-only spring-dashpot force,
- rigid-rigid contact using a direct pairwise normal-only spring-dashpot force,
- no tangential history,
- no rigid contact diagnostics in VTK/meta output,
- and no resting-contact projection fallback.

### `collision`

`collision` adds:

- a much richer runtime contact model,
- tangential spring-dashpot friction,
- material/specification models for wall contact,
- contact history updated through `UpdateCallback`,
- a resting-contact projection fallback,
- contact diagnostics and richer I/O,
- more rigid-wall tests and examples,
- but also removes rigid-rigid contact in the current `rhs.jl`.

## Target End State

After the stack lands on `main`, the intended behavior should be:

- rigid-rigid contact still works,
- rigid-rigid can use the shared runtime contact-law extensions where that makes sense,
- rigid-wall contact gains the richer wall-specific functionality from this branch,
- users get examples, tests, and docs together with the functionality that enables them,
- and compatibility cleanup is deferred until after the functional stack is stable.

## Recommended PR Stack

| PR | Title | Main Theme | Must Also Work For Rigid-Rigid? |
| --- | --- | --- | --- |
| 1 | Shared active rigid contact helpers and diagnostics | shared normal-force helpers, active runtime pieces, shared diagnostics | Yes |
| 2 | Shared frictional rigid contact and callback-driven history | tangential/friction runtime fields, history lifecycle, callback integration | Yes |
| 3 | Wall-specific manifold extensions and active wall controls | wall manifold state, patch normalization, and remaining wall-only active controls | No |
| 4 | Wall-only resting-contact projection | projection runtime field and persistent low-speed wall-contact fallback | No |
| 5 | Typed wall-contact specifications and user-facing examples | material/spec models, examples, docs, tests | Indirectly only |
| 6 | Optional cleanup/infrastructure | alias removal or infra-only refactors if needed | Depends |

The sections below describe each PR in detail.

---

## PR 1: Shared Active Rigid Contact Helpers And Diagnostics

### Proposed title

`Share active rigid contact helpers and diagnostics`

### Objective

Port only the active shared rigid-contact functionality needed immediately on `main`:

- shared normal-force helpers used by both rigid-wall and rigid-rigid contact,
- and shared rigid-contact diagnostics and their I/O exposure.

This PR should not introduce dormant advanced runtime parameters yet.
It establishes the minimal shared foundation that later PRs will build on.

### Why this PR exists separately

Right now, `collision` bundles active shared changes and future-facing runtime API in the
same area of the codebase.

Under the "no dormant API" rule, those should not be reviewed together.

This PR should answer one question only:

"Can `main` safely factor the currently active shared rigid-contact helpers and diagnostics
without breaking rigid-rigid contact?"

### Primary files in scope

- `src/schemes/structure/rigid_body/contact_models.jl`
- `src/schemes/structure/rigid_body/rhs.jl`
- `src/schemes/structure/rigid_body/system.jl`
- `src/io/io.jl`
- `src/io/write_vtk.jl`
- `test/systems/rigid_system.jl`
- `docs/src/systems/rigid_body.md`
- optionally `NEWS.md`

### Functional changes to port

Port the shared helper functions that become active in this PR:

- `normal_contact_force_components`
- `normal_contact_force`

These helpers should live in a place that both the wall and rigid-rigid paths can call.

Port the shared rigid-contact diagnostics:

- `contact_count`
- `max_contact_penetration`

Do not port friction/history or wall-only runtime parameters yet.
If a `RigidContactModel` change is not exercised by active PR1 behavior, defer it.

### Important design rule

Do not let the shared helper API assume "wall" or "manifold".

The helper layer should work with generic quantities:

- penetration,
- normal,
- normal velocity,
- tangential velocity,
- and tangential displacement.

### Rigid-rigid requirement

The rigid-rigid path in `rhs.jl` must remain active in this PR.

Specifically:

- keep `main`'s rigid-rigid `interact!`,
- refactor it to use the shared normal-force helper,
- and keep rigid-rigid normal-only in this PR.

The unacceptable state is:

- rigid-rigid becoming a no-op.

The acceptable interim state is:

- rigid-rigid still works,
- rigid-rigid still uses normal-only forces,
- shared normal-force logic is factored out,
- and no dormant tangential/friction API is introduced yet.

### Diagnostics to add in this PR

Port the corresponding output support for `contact_count` and `max_contact_penetration`:

- VTK fields in `src/io/write_vtk.jl`
- metadata export in `src/io/io.jl`
- system-data accessors in `src/schemes/structure/rigid_body/system.jl`

### Tests to include in this PR

Expand `test/systems/rigid_system.jl` with tests that cover:

- the active normal-only `RigidContactModel` behavior and validation,
- rigid-rigid normal-only contact still functioning,
- rigid-wall normal-only contact still functioning,
- `contact_count` and `max_contact_penetration` being exported and populated,
- and `system_data`/VTK-facing diagnostics exposing the new fields.

### Docs to include in this PR

Update `docs/src/systems/rigid_body.md` to document:

- the active normal-only rigid-contact behavior on `main`,
- the new shared diagnostics,
- and that richer friction/history and wall-only controls are introduced only in later PRs
  when they become active.

### Acceptance criteria

- `main` retains working rigid-rigid contact.
- The shared normal-contact helper layer is active for both rigid-wall and rigid-rigid
  normal contact.
- Shared diagnostics exist and are exposed through I/O.
- No dormant runtime/contact-law API is introduced.
- Existing normal-only behavior remains the default for users who do not opt into advanced
  parameters.

### Explicit non-goals

- no friction/tangential runtime parameters yet,
- no contact history yet,
- no `UpdateCallback` requirement yet,
- no wall-only control parameters yet,
- no wall-specification models yet,
- no resting-contact projection yet,
- no alias cleanup yet.

---

## PR 2: Shared Frictional Rigid Contact And Callback-Driven History

### Proposed title

`Add frictional rigid contact and shared history updates through UpdateCallback`

### Objective

Port frictional/tangential rigid contact as live behavior and hook its state updates into
`UpdateCallback`, but design it so it is not wall-only.

Under the "no dormant API" rule, this is also the PR where the tangential/friction runtime
fields should first appear, because this PR is what makes them active.

### Why this PR exists separately

The branch introduces richer tangential/friction laws and a callback-driven per-step
history update mechanism.

Without the "no dormant API" rule, those could be staged separately.
With that rule, they belong together: the runtime fields become user-facing only when the
callback/history implementation makes them real behavior.

### Primary files in scope

- `src/schemes/structure/rigid_body/contact.jl`
- `src/schemes/structure/rigid_body/contact_forces.jl`
- `src/callbacks/update.jl`
- `src/schemes/structure/rigid_body/system.jl`
- possibly `src/general/semidiscretization.jl` only if needed for callback checks
- `test/systems/rigid_system.jl`
- `docs/src/systems/rigid_body.md`
- `docs/src/callbacks.md`

### Functional changes to port

Introduce the friction/history-dependent runtime fields only when they are first consumed
by active code in this PR. At minimum this should include:

- `static_friction_coefficient`
- `kinetic_friction_coefficient`
- `tangential_stiffness`
- `tangential_damping`
- `stick_velocity_tolerance`
- and any penetration/slop parameter that the active friction/history law actually uses

Port the history-management pieces from the branch:

- `update_rigid_contact_eachstep!`
- active contact-key tracking
- inactive contact-pair cleanup
- tangential displacement updates
- and the callback hook added in `src/callbacks/update.jl`

Port the shared helper functions that first become live here:

- `normal_friction_reference_force`
- `tangential_contact_force`

If a helper or runtime field is still unused after this PR, defer it again.

### Required redesign for rigid-rigid support

The branch currently uses a wall-oriented contact-key scheme based on
`NTuple{3, Int}` keys tied to wall manifolds.

That is too wall-specific to become the shared foundation.

This PR should introduce a shared key strategy that supports both:

- wall contacts,
- and rigid-rigid contacts.

Recommended direction:

- define a small dedicated key type instead of raw tuples,
- or define two concrete key types and use a concrete union only if it remains type-stable.

A practical shared shape is:

- `neighbor_system_index`
- `local_particle`
- `contact_slot`
- `contact_kind`

where:

- `contact_slot` means manifold index for wall contacts,
- and `contact_slot` means neighbor particle index for rigid-rigid contacts.

The exact encoding is less important than the design rule:

the history container must not force wall-only semantics on rigid-rigid contact.

### Rigid-rigid requirement

By the end of this PR, the history infrastructure should be usable by rigid-rigid as well.

That means:

- rigid-rigid contact must still exist,
- rigid-rigid contacts must not be excluded from the history design,
- and if tangential friction is enabled for rigid-rigid, the rigid-rigid path should be able
  to store and update its tangential displacement state.

If full rigid-rigid tangential friction is too large for this PR, the minimum acceptable
outcome is:

- the history data model and update path are generic enough for rigid-rigid,
- the rigid-rigid path still works,
- and there is no architecture debt that would force a wall-only redesign later.

### Callback behavior to document

This PR is where the feature becomes callback-dependent.

Document clearly that `UpdateCallback` is required when a rigid contact model uses:

- tangential history-dependent terms, or
- later projection-based features introduced in the next PRs.

### Tests to include in this PR

Extend `test/systems/rigid_system.jl` with tests for:

- tangential history update and cleanup for wall contact,
- callback requirement checks,
- rigid-rigid contact paths remaining valid after the callback hook is introduced,
- and, if implemented here, rigid-rigid tangential-history persistence across updates.

Add regression tests ensuring:

- inactive contact pairs are removed,
- active pairs persist,
- and normal-only contact does not require unnecessary history state.

### Docs to include in this PR

Update:

- `docs/src/systems/rigid_body.md`
- `docs/src/callbacks.md`

to explain:

- why `UpdateCallback` is needed,
- which rigid-contact features depend on it,
- and how users should configure examples/simulations accordingly.

If an existing example is updated to use frictional rigid contact, update it in this PR and
make the callback requirement explicit there as well.

### Acceptance criteria

- The friction/history-dependent runtime fields introduced here are active in code that
  lands in this same PR.
- The rigid-contact history lifecycle is callback-driven.
- The history design is generic enough for rigid-rigid contact.
- Rigid-rigid contact is not broken or silently excluded.
- No dormant friction/history API remains after the PR is merged.

### Explicit non-goals

- no wall-specific patch normalization yet,
- no wall-only resting-contact projection yet,
- no wall-only control parameters that are still inactive,
- no material/specification models yet.

---

## PR 3: Wall-Specific Manifold Extensions And Active Wall Controls

### Proposed title

`Extend rigid-wall manifolds and activate wall-only controls`

### Objective

Port the wall-specific parts of the branch that improve rigid-wall contact quality but do
not naturally apply to rigid-rigid contact, and activate any remaining wall-only runtime
controls whose behavior first becomes live here.

### Why this PR exists separately

The current `main` wall path already has manifold reduction.

This PR should review only the incremental wall-specific refinements:

- extra wall-manifold scratch state,
- patch clustering / normalization,
- and the wall-specific controls that become active together with that logic.

### Primary files in scope

- `src/schemes/structure/rigid_body/rhs.jl`
- `src/schemes/structure/rigid_body/system.jl`
- `test/systems/rigid_system.jl`
- `docs/src/systems/rigid_body.md`

### Functional changes to port

Port the wall-specific cache and logic:

- patch clustering buffers,
- wall-manifold extensions needed by the branch implementation,
- `normalize_force_by_contact_patch`,
- `torque_free` if it is activated here,
- and any wall-only data reductions that improve resolution dependence.

As with the rest of the stack, do not introduce wall-only knobs here unless they become
active in this PR.

### Rigid-rigid requirement

This PR must not redefine rigid-rigid behavior.

Specifically:

- do not move rigid-rigid onto wall-manifold abstractions,
- do not delete the rigid-rigid `interact!`,
- and do not let wall-specific cache layout leak into the rigid-rigid path.

### Tests to include in this PR

Extend `test/systems/rigid_system.jl` with wall-focused tests for:

- patch normalization on/off behavior,
- wall-contact force aggregation staying bounded and resolution-tolerant,
- and diagnostic values such as `contact_count` and `max_contact_penetration`.

### Docs to include in this PR

Update `docs/src/systems/rigid_body.md` to clarify:

- which parts of the contact stack are wall-specific,
- what `normalize_force_by_contact_patch` does,
- what any other newly active wall-only controls do,
- and when a user should enable or leave them disabled.

### Acceptance criteria

- Wall-specific quality improvements land without changing the rigid-rigid contract.
- The wall-only nature of the feature is explicit in code and docs.
- No dormant wall-only runtime option is introduced.

### Explicit non-goals

- no wall-specification models yet,
- no resting-contact projection yet,
- no alias cleanup yet.

---

## PR 4: Wall-Only Resting-Contact Projection

### Proposed title

`Add resting-contact projection fallback for rigid-wall contact`

### Objective

Port the low-speed wall-contact projection fallback from the branch.

This is the mechanism that prevents rigid bodies from remaining in penetrated or numerically
stuck wall-contact states when adaptive timesteps collapse in the settled-contact regime.

Under the "no dormant API" rule, this is also the PR where `resting_contact_projection`
should first appear as a user-facing runtime option.

### Why this PR exists separately

Projection is a numerical stabilization feature with its own review burden:

- it changes state outside the ordinary force evaluation path,
- it depends on callback timing,
- and it is currently built specifically around wall-manifold reconstruction.

That should not be hidden inside a larger contact PR.

### Primary files in scope

- `src/schemes/structure/rigid_body/contact_projection.jl`
- `src/schemes/structure/rigid_body/contact.jl`
- `src/callbacks/update.jl`
- `test/systems/rigid_system.jl`
- `docs/src/systems/rigid_body.md`
- examples that require the projection path, if any are added or modified

### Functional changes to port

Port:

- the `resting_contact_projection` runtime field,
- persistence counters,
- resting-contact detection,
- projection trigger logic,
- position/velocity correction,
- and projected-history cleanup.

### Rigid-rigid requirement

This PR is explicitly wall-only unless a separate, correct rigid-rigid projection design is
also introduced.

Do not attempt to "generalize" the current branch implementation to rigid-rigid in a rushed
way. The current projection logic is built around wall constraints and wall-manifold data.

The requirement for rigid-rigid here is:

- rigid-rigid contact must continue to function,
- rigid-rigid must not accidentally start using wall-only projection logic,
- and the wall-only scope must be clear in code and docs.

### Tests to include in this PR

Extend `test/systems/rigid_system.jl` with:

- the projection fallback tests from the branch,
- trigger-threshold tests,
- persistence counter tests,
- and explicit tests that projection does not activate on the first low-speed impact step.

If possible, add one guard test that confirms rigid-rigid simulations are unaffected by the
wall-only projection implementation.

### Docs to include in this PR

Update `docs/src/systems/rigid_body.md` to explain:

- what resting-contact projection is,
- why it exists,
- that it is currently for rigid-wall contact,
- and that it depends on `UpdateCallback`.

### Acceptance criteria

- The branch's wall-settling stabilization is available on `main`.
- Its wall-only scope is explicit.
- It does not alter rigid-rigid behavior.
- No projection-only control lands before this PR.

### Explicit non-goals

- no material/specification models yet,
- no API cleanup yet.

---

## PR 5: Typed Wall-Contact Specifications, Examples, Docs, And Coverage

### Proposed title

`Add typed rigid-wall contact specifications and material-driven examples`

### Objective

Port the user-facing wall-contact specification models and ship the corresponding examples,
tests, and documentation in the same PR.

### Why this PR exists separately

This PR is the user-facing layer built on top of the earlier runtime work.

It should not be merged until the runtime and callback foundations are stable.

### Primary files in scope

- `src/schemes/structure/rigid_body/contact_models.jl`
- `src/TrixiParticles.jl`
- `examples/structure/falling_rigid_sphere_2d.jl`
- `examples/structure/perfect_elastic_sphere_wall_2d.jl`
- `examples/structure/falling_steel_spheres_analytical_2d.jl`
- `test/examples/examples.jl`
- `test/systems/rigid_system.jl`
- `docs/src/systems/boundary.md`
- `docs/src/systems/rigid_body.md`
- optionally `docs/src/examples.md`
- optionally `NEWS.md`

### Functional changes to port

Port the branch's typed wall-contact specification models:

- `PerfectElasticBoundaryContactModel`
- `LinearizedHertzMindlinBoundaryContactModel`
- `copy_contact_model`

Export them publicly from `src/TrixiParticles.jl`.

### Scope rule

These models are wall-contact specification types.

Do not silently reinterpret them as generic rigid-rigid material models in this PR.
If later you want material-driven rigid-rigid contact, that should be designed as its own
follow-up feature with its own API.

### Examples to include in this PR

Ship the examples together with the enabling functionality:

- `examples/structure/falling_rigid_sphere_2d.jl`
- `examples/structure/perfect_elastic_sphere_wall_2d.jl`
- `examples/structure/falling_steel_spheres_analytical_2d.jl`

If any existing example must change because of callback requirements or renamed API
arguments, update it here rather than in a later cleanup PR.

### Tests to include in this PR

Add and update:

- material/specification conversion tests in `test/systems/rigid_system.jl`,
- example tests in `test/examples/examples.jl`,
- and any extra test dependencies only if the new tests truly need them.

The test set should cover:

- conversion from specification models to runtime `RigidContactModel`,
- basic invariants of the generated runtime parameters,
- example execution,
- and analytical/regression behavior for the new structure examples.

### Docs to include in this PR

Update:

- `docs/src/systems/boundary.md`
- `docs/src/systems/rigid_body.md`
- optionally `docs/src/examples.md`

to describe:

- the new typed wall-contact models,
- when to use runtime `RigidContactModel` directly versus a specification model,
- and which examples demonstrate which features.

### Acceptance criteria

- The new specification models are fully user-facing.
- Examples, docs, and tests ship with the models that enable them.
- No separate docs/examples follow-up is needed for the material-driven wall-contact path.

### Explicit non-goals

- no generic material-driven rigid-rigid model yet,
- no compatibility alias removal yet unless it becomes unavoidable.

---

## Optional PR 6: Cleanup And Infrastructure Only If Needed

### Possible titles

- `Remove rigid boundary contact compatibility aliases`
- `Refactor Semidiscretization Adapt paths`
- `Cleanup post-port rigid contact API`

### Objective

Handle non-functional cleanup only after the functional stack has landed and stabilized.

### Candidate items

- remove `boundary_contact_model` compatibility aliases,
- remove old internal naming if still present,
- port the tuple-based `Semidiscretization`/`Adapt` refactors if they are actually needed,
- and move any purely mechanical renaming into a cleanup-only PR.

### Scope rule

Do not let this PR become a dumping ground.

If the functional stack works without an infrastructure refactor, leave the refactor out.

### Rigid-rigid requirement

No cleanup PR may weaken or silently alter rigid-rigid behavior.

---

## Cross-Cutting Engineering Decisions

### Decision A: Keep `main`'s rigid-rigid timestep logic

`main` computes rigid contact timestep limits with awareness of rigid-rigid neighbors.

The branch introduces a simplified single-system `contact_time_step(system)` path for the new
callback-driven wall logic.

When porting:

- preserve `main`'s pairwise rigid-rigid timestep logic for `calculate_dt`,
- and, if the branch's single-system helper is useful for wall-specific callback logic,
  keep it as an additional helper rather than a replacement.

### Decision B: Shared diagnostics mean shared semantics

If `contact_count` and `max_contact_penetration` are exposed as generic rigid-body
diagnostics, define them consistently:

- `contact_count`: total accepted contact contributions for the rigid system during the
  current update window,
- `max_contact_penetration`: maximum effective penetration seen for that rigid system during
  the current update window,

including both wall and rigid-rigid contacts where applicable.

### Decision C: Delay alias removal

The branch contains API cleanup around `boundary_contact_model`.

That cleanup is not required to land the functionality. The lower-risk path is:

- keep compatibility aliases during the functional port,
- switch examples/docs to the preferred API,
- remove aliases only in a later cleanup PR if desired.

### Decision D: Avoid wall-only abstractions in shared code

Any code intended to support rigid-rigid and rigid-wall contact should avoid names and data
structures that encode wall semantics by construction.

Examples of wall-only concepts:

- manifold index,
- wall velocity sum,
- patch cluster,
- resting wall constraint.

Examples of shared concepts:

- penetration,
- contact normal,
- normal velocity,
- tangential velocity,
- tangential displacement history,
- contact-law parameters.

### Decision E: Runtime Fields Land Only When They Become Active

Do not introduce `RigidContactModel` parameters ahead of the PR that makes them real
behavior.

Concretely:

- keep PR1 limited to active normal-contact helpers and diagnostics,
- introduce friction/history fields with PR2,
- introduce remaining wall-only controls with PR3 and PR4,
- and update examples only when those options are actually usable.

## Suggested Branch Stack

If you want to implement this as a stacked series, a reasonable branch naming scheme is:

- `port/rigid-contact-runtime`
- `port/rigid-contact-history`
- `port/rigid-wall-manifolds`
- `port/rigid-wall-projection`
- `port/rigid-wall-spec-models`
- optional `port/rigid-contact-cleanup`

Each branch should target the previous one until the stack is ready, then be merged from the
bottom up.

## Recommended Review Strategy

For each PR, reviewers should explicitly check:

1. Does rigid-rigid still work?
2. Is the PR only introducing the feature claimed in its title?
3. Are docs/examples/tests included only where they are functionally justified?
4. Is any wall-only logic incorrectly leaking into shared code?
5. Is any shared functionality incorrectly scoped as wall-only?

## Final Recommendation

Do not port this branch by following the branch commit history.

Port it by architectural slice:

- shared active helpers and diagnostics first,
- friction/history runtime fields together with callback plumbing second,
- wall-specific reductions and active wall controls third,
- wall-only projection fourth,
- user-facing typed wall models last,
- and cleanup only after the functional stack is stable.

That sequence gives the best chance of landing the rigid-wall improvements from this branch
without regressing `main`'s still-working rigid-rigid implementation.

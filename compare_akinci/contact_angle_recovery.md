# Contact-Angle Recovery Design

This note records the Phase 3 recovery diagnostics after both implemented contact-angle candidates
failed the off-target restoring gate. It is intentionally separate from the accepted Phase 2 data.
No formulation below is a production recommendation until it passes the gates in `CSS_plan.md`.

## Requirements

A replacement must:

- preserve `ColorfieldSurfaceNormal(contact_model=nothing)` behavior;
- use physical surface tension in N/m without a fitted angle-dependent coefficient;
- give the correct control-subtracted restoring sign for all four Phase 3 cap pairs;
- retain pairwise-conservative CSS fluid stress and zero wall-normal contact force;
- define a contact-line measure whose volume integral converges to physical line length;
- remain finite at 0 and 180 degrees and give exactly zero wall-energy force at 90 degrees;
- pass the five-angle, three-resolution static matrix and timestep/stability gates.

## Rejected Direct Corrections

### Boundary-subtracted CLF normal

Before normalization, dummy-boundary quadrature contributes the same vector to the total surface
normal and cached wall normal, so the fluid-only candidate can be reconstructed as

```math
\bm n_\mathrm{fluid}=\bm n_\mathrm{total}-\bm n_\mathrm{wall}.
```

This is not a usable dynamic-angle estimator. At 1500 particles its maximum target-initialized mean
angle error is 58.5 degrees and as much as 85% of contact-line weight has the wrong restoring sign.
The wall-completed normal is also invalid as a shape-angle estimator: its maximum static error is
54.1 degrees. Adding a CLF contact-normal cache or blending these vectors is therefore rejected.

Evidence: `validation/surface_tension_3d/contact_angle_normal_components.csv`.

### Ghost-gradient geometric rotation

A reconstruction variant retained the raw tangential gradient ``\bm q`` and imposed the target wall
component,

```math
\bm n_\theta=\bm q+\lVert\bm q\rVert\cot\theta\,\hat{\bm n}_w,
```

then recomputed the one-sided surface delta from ``\lVert\bm n_\theta\rVert``. This is more
consistent than rotating a fixed-magnitude normal, but it reproduces the same two-of-four
fixed-particle sign result as the existing geometric model. The issue is not only normal magnitude
or tangent selection; no geometric production patch is justified.

Evidence: `validation/surface_tension_3d/contact_angle_force_sign_ghost_geometric.csv`.

## Wall Free-Energy Formulation

The remaining physically motivated path is a discrete wall-energy force. The CSS surface-stress
divergence already supplies the free-surface line tension. Varying the solid-liquid and solid-gas
wall energies adds the Young contribution

```math
\bm a_a^w
=\frac{\sigma}{\rho_a}\cos\theta_e\,\delta_{CL,a}\,\hat{\bm t}_{w,a}.
```

This force needs the equilibrium target only; it does not subtract a noisy particle estimate of the
dynamic angle. Consequently it is exactly zero at 90 degrees. On the four fixed Phase 3 caps, the
target-only force with the current line delta gives three correct total acceleration signs. Applying
the one-phase factor of two gives four of four. This factor is theoretically plausible because the
free surface is sampled from only one side, but force-sign success is not sufficient evidence for
normalization.

Evidence: `validation/surface_tension_3d/contact_angle_force_sign_wall_energy.csv`.

## Contact-Line Measure Blocker

For a spherical cap with contact radius ``r_c``, the line measure must satisfy

```math
L_h=\sum_a \frac{m_a}{\rho_a}\delta_{CL,a}
\longrightarrow 2\pi r_c.
```

The current divergence-based line delta is 24-77% low over the five-angle, three-resolution matrix
and becomes worse under refinement for several obtuse cases. A coarea candidate based on
``\lVert\nabla c_s\times\nabla c_w\rVert`` also fails the 20% gate after the standard one-sided
factors; errors are 22-41%. Omitting interface activity does not restore convergence. A numerical
factor close to ``2\pi`` reduces these errors, but it has not been derived for arbitrary kernels and
would currently be an empirical multiplier, so it is not admissible.

## Kernel-Level Normalization Study

For a radial kernel with compact-support radius ``R``, let ``g_W(d)`` be the magnitude of the
half-space color gradient represented by the implemented `kernel_grad` at distance ``d`` from a
plane. Direct integration gives

```math
g_W(d)=2\pi\int_d^R W(r)r\,\mathrm{d}r
-\pi(R^2-d^2)W(R^-),
\qquad
J_W=\int_0^R g_W(d)\,\mathrm{d}d.
```

The cutoff term matters because `kernel_grad` differentiates only inside the strict support. It is
zero for kernels that vanish at ``R``. It is not zero for the truncated Laguerre-Gauss kernel, whose
implemented gradient half-integral is 24.9% below one half of its nominal kernel mass.

For two planes intersecting at angle ``\theta``, the cross-gradient contributes
``\sin\theta\,g_W(s)g_W(w)`` while the coordinate Jacobian contributes
``1/\sin\theta``. Integrating only the represented fluid quadrant therefore gives ``J_W^2`` per
unit contact-line length, independent of angle. The kernel-derived coarea normalization is

```math
C_W=\frac{1}{J_W^2}.
```

For normalized kernels that vanish at their support boundary, ``J_W=1/2`` and ``C_W=4``. No
sessile-drop angle or measured line length enters this factor.

The `line_normalization` mode evaluates this continuous profile and matching three-dimensional
kernel-gradient lattice sums for all ten smoothing kernels, five angles, and
``h/\Delta x\in\{2,4,8\}``. Translational invariance collapses the periodic contact-line direction
to unit length; four interface/lattice phases are averaged. The resulting 150 rows show:

- coarea passes the 20% middle-resolution gate in 50/50 kernel/angle cases, with a maximum error of
  3.37%;
- the strict endpoint-decrease gate passes 40/50 cases; the ten non-monotone fine-grid errors are all
  at most 0.105%, but the frozen gate is not relaxed after observing them;
- the production-style divergence form, normalized by its orthogonal kernel integral, passes only
  9/50 middle-resolution cases and no non-orthogonal case;
- for the production Wendland C2 kernel, coarea passes all five planar middle and endpoint gates.

Evidence: `validation/surface_tension_3d/contact_line_normalization.csv`.

## Spherical-Cap Transfer And Decision

The successful planar Wendland C2 factor is ``C_W=4``, which is exactly the previously tested raw
coarea factor. On the actual 1500-particle spherical caps its five line-length errors are 25.8%,
22.4%, 28.5%, 34.3%, and 34.2%. Dividing by the existing local reproducing/support moment reduces
them to 3.4%, 3.2%, 9.9%, 21.0%, and 24.1%, but the two obtuse cases still fail and worsen under
refinement. Thus the continuum normalization is derived, but the available discrete wall/interface
gradients do not transfer it to curved sessile geometry with the required convergence.

This initial line-measure gate did not close, so no wall-free-energy force entered production.
`contact_model=nothing` remained the default while the validation-only continuation below tested
compatible indicators and a wetted-area discretization under the same measure-first rule. Akinci
adhesion and per-angle force gains remained excluded.

## R6 Continuation: Three-Way Comparison

Three candidates share one comparison table and measure-first gates. All live as validation-only
modes of `validation/surface_tension_3d/contact_angle_decision.jl`; no production cache or force
changes before a measure gate passes on caps. Uniform-lattice volume weighting is excluded as a
cause by construction (`V_a = V_b` on the initial lattice).

### R6-D: Cap-transfer attribution diagnostic (runs first)

A derived fact fixes what the cap coarea measure actually sees: `calc_boundary_normal!` adds the
identical threshold-gated, `V_a`-weighted completion vector to `surface_normal` and
`boundary_normal`, so

```math
\bm n_\mathrm{total}\times\bm n_\mathrm{wall}
=\bm n_\mathrm{fluid}\times\bm n_\mathrm{wall}.
```

The cross product therefore pairs a wedge-restricted fluid gradient with a gated wall proxy, while
the passing canonical study paired full half-space indicator gradients. The diagnostic closes this
gap from both sides, one ingredient at a time:

- planar variants added to `line_normalization`: (a) wedge-restricted interface gradient
  (neighbors require both `wall_distance > 0` and the fluid side of the interface); (b) wall
  gradient gated by the lattice colorfield at `0.1` of its maximum, mirroring
  `boundary_contact_threshold`;
- cap variants in a new `cap_transfer` mode: substitute the canonical profile ``g_W`` for (a) the
  discrete wall gradient at each particle's wall distance, (b) the discrete interface gradient at
  each particle's distance to the analytic cap surface, (c) both, isolating lattice quadrature and
  contact-line curvature.

The planar study now reproduces the transfer loss. Across ten kernels and five angles, the ideal
coarea middle gate remains 50/50; fluid-wedge restriction reduces it to 20/50, and adding the 0.1
wall-colorfield gate reduces it to 16/50. For Wendland C2 specifically, only the 120- and 150-degree
middle cases pass after either production ingredient is applied. The gated middle errors are 47.0%,
38.2%, 29.0%, 19.9%, and 12.3%.

The cap substitutions produce:

| Variant | Middle passes | Endpoint passes | Maximum middle error |
|---|---:|---:|---:|
| production discrete | 0/5 | 2/5 | 34.3% |
| analytic wall only | 3/5 | 2/5 | 26.2% |
| analytic interface only | 4/5 | 1/5 | 30.0% |
| analytic wall and interface | 5/5 | 1/5 | 16.1% |
| support-moment diagnostic | 3/5 | 1/5 | 24.1% |

Thus both discrete fields contribute, with the missing compatible interface continuation explaining
the larger share of middle-resolution failures. Even replacing both fields leaves non-monotone
curved-cap quadrature in four angles, so no scalar normalization can satisfy the frozen endpoint
gate. Evidence: `contact_line_normalization.csv` and `contact_line_cap_transfer.csv`.

### R6-C: Compatible colorfield continuation

The implemented validation reconstruction removes the hard gate from the wall field and continues
the fluid indicator into wall particles with their flooded-reference-normalized wetness
``\phi_b=c_b/\max(c)``:

```math
\nabla c_w=\sum_{b\in wall}V_a\nabla W_{ab},\qquad
\nabla c_s=\nabla c_\mathrm{fluid}+\sum_{b\in wall}V_a\phi_b\nabla W_{ab}.
```

This is a compatible reference-normalized colorfield continuation, not a local Shepard quotient;
the name was corrected after implementation. It uses the existing boundary colorfield and one
validation vector accumulation. The derived factor ``C_W`` is unchanged and no fitted constant is
introduced.

All five middle cap errors pass, with maximum 8.85%, but all five strict endpoint-decrease gates
fail: fine errors are 4.9-17.3% after exceptionally small and non-monotone coarse errors. R6-C is
therefore ineligible and no force-sign or ODE run counts for it.

### R6-W: Wetted-area wall free energy

Young-Dupre gives the wall energy without any contact-line delta:

```math
E_w=(\sigma_{sl}-\sigma_{sg})A_{sl}=-\sigma\cos\theta_e\,A_\mathrm{wetted}.
```

Discretize the wetted area through the boundary colorfield that production already computes,
restricted to the exposed wall layer with per-particle area ``A_b=\Delta x^2``:

```math
A_\mathrm{wetted}=\sum_b A_b\,H\!\left(c_b/c_\mathrm{ref}\right),
\qquad
c_b=\sum_a\frac{m_a}{\rho_a}W_{ab},
```

with ``H`` the cubic smoothstep already used for activity transitions over the full normalized
range (no free width parameter). The validation uses the maximum current boundary colorfield as
``c_\mathrm{ref}`` and explicitly restricts the sum to the exposed wall layer. A production version
would need a local or precomputed flooded-wall reference instead of this global reduction.
Differentiating the energy at frozen densities yields the fluid acceleration

```math
\bm a_a=\frac{\sigma\cos\theta_e}{\rho_a}
\sum_b\frac{A_b}{c_\mathrm{ref}}\,H'\!\left(c_b/c_\mathrm{ref}\right)\nabla_a W_{ab}.
```

Units are `m/s^2` (``\sigma A_b\nabla W/(\rho_a c_\mathrm{ref})`` carries N per kg); the force is
exactly zero at 90 degrees and finite at 0/180, while ``H'`` localizes it at the contact line. The
validation computes only the fluid-side force and neglects the density-gradient term of
``\partial c_b/\partial\bm x_a``. A production version would also need an equal/opposite wall
reaction. Measure gate before any force test:
``|A_h/(\pi r_c^2)-1|`` at most 20% at 1500 particles with decreasing endpoint error over all five
angles; then exact zero at 90 degrees, then the four fixed-particle signs. Evidence:
`wetted_area_measure.csv` and `contact_angle_force_sign_wetted_area.csv`.

R6-W passes four of five middle and endpoint area gates. The unresolved 150-degree area is 50.6%
high at 1500 particles and remains 39.4% high at 3000 particles because its small physical contact
disk is comparable to the smoothing transition. The wall acceleration is exactly zero for both
90-degree-target cases and total fixed-particle signs are 4/4, but those signs do not make the model
eligible after the area gate fails.

### Selection

No candidate passes all five middle and endpoint measure gates. Consequently no R6 candidate
advances to dynamics, `contact_model=nothing` remains the default, G3 stays open, and Phase 4 wetting
work remains blocked. The compact decision table is
`validation/surface_tension_3d/contact_angle_recovery_comparison.csv`.

## R7 Pre-Registered Continuation

The strict R6 endpoint rule rejects the exact-profile cap control in four of five angles. R7 first
tests whether this is lattice-phase quadrature rather than formulation error. Every new cap measure
is averaged over eight rank-1 horizontal phases, while the wall-normal phase and the production
`h/dx=1.4` remain fixed. A discarded four-phase diagonal control had zero useful variance because
all samples were related by square-lattice reflection symmetry; no candidate had run when this was
corrected. The 20% middle gate is unchanged. A fine endpoint passes only when it is within 20% and
is no more than two combined phase standard errors worse than the coarse endpoint.

An independent exact-profile series holds physical `h` fixed and increases `h/dx` through
`{1.4, 2.8, 4.2}`. This converges to the exact profiles' nonzero curvature-smoothing bias, not zero;
the control therefore requires the fine-to-middle change not to exceed the middle-to-coarse change
by more than two combined phase standard errors and a fine error below 20%. This control-only
correction and the final phase set were recorded before
generating any R7 candidate evidence.

Three validation-only formulations then run regardless of earlier outcomes:

1. The wetted-area energy uses the kernel half-space convolution as its flooded reference. Its edge
   correction is the integral displacement of the canonical wedge profile after applying the same
   cubic smoothstep as the discrete area. Differentiating the corrected radius supplies the force
   chain-rule factor; no cap result is fitted.
2. The compatible continuation is paired with the exact plate normal and canonical wall profile,
   completing the missing R6-C geometry-normal combination.
3. A Young color boundary condition extrapolates exposed-wall wetness into ghost layers with the
   prescribed normal derivative before the SPH gradient is assembled. Its clamped 0/180-degree
   limits remain finite. This is not the rejected post-assembly normal rotation.

R7 writes new evidence and leaves every production contact model and default unchanged.

## R7 Results

The exact-profile control validates the amended protocol. Its production series passes all five
middle and uncertainty-aware endpoint checks. At fixed physical smoothing length, all five
`h/dx={1.4,2.8,4.2}` series remain within 20% and satisfy the uncertainty-aware Cauchy check. The
eight phases expose real lattice variance while preserving the fluid-wall gap.

### Corrected wetted-area energy

The kernel half-space reference is about 7.69% above the maximum colorfield found on these finite
caps. The canonical wedge calculation gives edge displacements in units of `h` of approximately
`{-0.5072,-0.1747,0,0.1747,0.5072}` at `{30,60,90,120,150}` degrees. Applying the displacement to
the effective wetted radius and differentiating that correction gives:

| Target | Middle corrected area error |
|---:|---:|
| 30 deg | 5.68% |
| 60 deg | 4.14% |
| 90 deg | 3.66% |
| 120 deg | 2.77% |
| 150 deg | 1.99% |

All five middle and endpoint gates pass. The corrected energy force gives `4/4` correctly directed
total fixed-cap accelerations and is exactly zero for both 90-degree-target cases. R7-W is therefore
the only candidate eligible to enter R4 dynamics. This does not yet justify a production
implementation: the dynamic replay, full static matrix, equal/opposite wall reaction, and treatment
of the density dependence in the colorfield derivative remain unresolved.

### Compatible geometry wall

Pairing compatible continuation with the exact plate normal and canonical wall profile passes all
five middle line-length errors but only two endpoint gates. Its line-weighted reconstructed angles
pass none of the five middle or endpoint angle checks. Its target-only wall energy has `4/4` total
fixed-cap signs, but measure-first eligibility fails, so R7-CG is rejected.

### Young color boundary

The scalar ghost continuation is finite and bounded in `[0,1]`. It passes all five middle and
endpoint line-measure checks, showing that it produces a coherent contact-line integral. It does not
impose the requested normal accurately enough: only two middle and one endpoint angle checks pass.
It also gives only `3/4` total and `1/4` contact-induced fixed-cap signs. R7-Y is rejected; passing a
line integral alone is not sufficient evidence for a wetting boundary condition.

Evidence:

- `validation/surface_tension_3d/contact_measure_protocol.csv`
- `validation/surface_tension_3d/wetted_area_corrected.csv`
- `validation/surface_tension_3d/contact_angle_recovery_extended.csv`
- `validation/surface_tension_3d/contact_angle_force_sign_extended.csv`
- `validation/surface_tension_3d/contact_angle_recovery_extended_comparison.csv`

No production source, cache, API, or default changed in R7. `contact_model=nothing` remained the
constructor default while R7-W advanced to the R4 work below.

## R4-W Pre-registration

R4-W keeps the corrected wetted-area model validation-only. The implementation is loaded by the
sessile-drop workbench and extends the existing contact hooks without changing `src/`, exported
types, or constructor defaults. The following discrete energy and gates are frozen before any R4-W
dynamic result is generated.

For exposed top-layer wall particles `b`, define

```math
c_b=\sum_a \frac{m_a}{\rho_a}W_{ab},\qquad
\xi_b=\operatorname{clamp}(c_b/c_\infty,0,1),\qquad
A_0=\Delta A\sum_b s(\xi_b),
```

where `s(x)=x^2(3-2x)`, `Delta A=dx^2`, and `c_inf` is the kernel half-space convolution at the
exposed-layer depth for the reference particle volume. `c_inf` is fixed during a run; only the
actual particle volumes in `c_b` carry density dependence. With the precomputed canonical R7 edge
displacement `d_theta`,

```math
r_0=\sqrt{A_0/\pi},\qquad
r=\max(r_0-hd_\theta,0),\qquad
A_h=\pi r^2,\qquad
E_h=-\sigma\cos(\theta)A_h.
```

No measured cap radius, fitted gain, angle offset, or case-dependent coefficient enters this
energy. The canonical wedge correction is singular at exactly 0 and 180 degrees, so this
validation model deliberately accepts only open-interval targets; those limits require a separate
derivation before any production proposal.

For `r_0>0` and `r>0`, let `g=dA_h/dA_0=r/r_0` and

```math
\lambda_b=\frac{g\Delta A}{c_\infty}s'(\xi_b),\qquad
S_a=\sum_b\lambda_bW_{ab},\qquad
q_a=\frac{\sigma\cos(\theta)}{\rho_a^2}S_a.
```

The derivative of the clamp is zero outside its open interval. The complete acceleration is the
sum of the explicit fluid-wall derivative and the density derivative,

```math
\boldsymbol a_a^{wall}
=\frac{\sigma\cos(\theta)}{\rho_a}
  \sum_b\lambda_b\nabla_a W_{ab},
\qquad
\boldsymbol a_a^{rho}
=-\sum_c m_c\left(q_a\frac{\rho_a}{\rho_c}
                    +q_c\frac{\rho_c}{\rho_a}\right)\nabla_a W_{ac}.
```

Each explicit pair contribution is accumulated with an equal and opposite wall force. The
fluid-fluid density term uses the symmetric pressure operator paired with the workbench's
`ContinuityDensity` equation and must have zero resultant by itself. Equivalently, its auxiliary
pressure is `p_a^*=rho_a^2 q_a=sigma*cos(theta)*S_a`. At 90 degrees the energy, both acceleration
terms, and wall reaction are exactly zero.

The staged gates are:

1. Algebra/static gate: finite energy and forces; directional energy-gradient agreement within
   `1e-5` relative error; explicit fluid-wall reaction residual and density-term resultant at most
   `1e-12` relative to their force scales; exact zero at 90 degrees; and `4/4` restoring total-force
   signs for the established 1500-particle off-target caps.
2. Perturbation gate: the unchanged `t=0.01 s`, damping `4000 s^-1`, threshold `0.1`, and
   1500-particle protocol must pass all four complete responses against fresh no-contact controls.
   If all acceleration signs pass but angle motion is below estimator resolution, exactly one
   uniform final-time extension may be applied to all four cases.
3. Static replay gate: all five target-initialized threshold-0.1 cases must remain within 5 degrees,
   settled, penetration-free, within `980--1020 kg/m^3`, and below 25% rejected steps.
4. Efficiency gate: rerun the existing `(90 degrees,1500)` and `(30 degrees,3000)` timestep cases,
   requiring `eta_p01>=0.05` and tail/head median `eta>=0.5`, then record three interleaved
   no-contact/model cost repetitions and the validation-cache bytes.
5. Full replay gate: only after gates 1--4 pass, run the 15-cell `{750,1500,3000}` matrix and the
   existing threshold/damping sensitivity matrix. G3 remains open until every Phase 3 completion
   condition passes.

Each stage writes a new `*_r4_wetted_area.csv` file and never replaces evidence for the rejected
geometric or contact-line-force models. Execution stops at the first failed stage.

### R4-W perturbation protocol correction

The first `0.01 s` run exposed a logical error in the inherited CLF response classifier, not a
force-formulation choice. It required every candidate to beat the no-contact control and to have a
nonzero contact-induced acceleration. For `theta=90 degrees`, however, the pre-registered energy is
identically zero, so R4-W and no contact must be exactly equivalent. Those two requirements were
therefore impossible by construction and contradicted the exact-zero gate above.

The original CSV is preserved. Before any longer run, the classifier is corrected as follows:

- non-90-degree targets retain the strict contact-induced acceleration and beats-control tests;
- 90-degree targets require zero contact-induced acceleration, exact equality with the no-contact
  trajectory, and a correctly directed **total** CSS acceleration;
- all safety and reaction gates remain unchanged.

With this formulation-consistent classifier, the initial run has four correctly directed effective
accelerations and three resolved motions; the unresolved motion is below 1 degree. This satisfies
the pre-registered condition for one uniform extension. The sole extension is frozen at `0.02 s`
for all four candidate/control pairs and writes a separate `*_extended.csv`; no initial evidence is
overwritten.

## R4-W Results

The complete density-dependent energy derivative passes every validation-only R4-W gate. The five
target-initialized algebra cases and four off-target force cases pass `9/9`; the worst directional
energy-gradient error is `4.89e-10`, versus the pre-registered `1e-5` tolerance. The largest initial
reaction/resultant residual is `1.22e-15`, and the 90-degree energy, acceleration, and reaction are
exactly zero.

The preserved initial perturbation CSV reports `2/4` under the contradictory inherited CLF
classifier. The formulation-consistent reclassification gives `4/4` effective acceleration signs
and `3/4` resolved motions. The single uniform `0.02 s` extension then passes all `4/4` complete
responses. Both 90-degree candidate trajectories are exactly equal to their no-contact controls, as
required by the zero Young wall energy.

The remaining staged results are:

| Gate | Result |
|---|---:|
| Threshold-0.1 target replay | `5/5` |
| Representative timestep cases | `2/2` |
| Repeated median runtime overhead | `2.0%` at zero-force 90 deg; `30.5%` at active 60 deg |
| Selected `{750,1500,3000}` matrix | `15/15` |
| Threshold/damping sensitivity | `4/4`; `0.107 deg` span |

The inherited repeated-cost case uses a 90-degree target, where this formulation is exactly
disabled, so its 2.0% overhead is only a zero-force-path measurement. Before production promotion,
the same interleaved protocol was repeated at an active 60-degree target and records 30.5% median
overhead. Cost was pre-registered as a measurement rather than a hard acceptance threshold; both
files are retained.

Across the selected matrix, the largest final circle-angle error is `4.33 deg`, density remains in
`[997.07,1000.02] kg/m^3`, maximum RMS speed is `4.32e-3 m/s`, and maximum rejected-step fraction is
15.2%. The largest recorded total momentum residual is `4.03e-15`. No target-dependent gain,
measured cap radius, angle offset, or case-specific damping was introduced.

Evidence:

- `validation/surface_tension_3d/contact_angle_static_r4_wetted_area.csv`
- `validation/surface_tension_3d/contact_angle_perturbation_r4_wetted_area.csv`
- `validation/surface_tension_3d/contact_angle_perturbation_r4_wetted_area_classified.csv`
- `validation/surface_tension_3d/contact_angle_perturbation_r4_wetted_area_extended.csv`
- `validation/surface_tension_3d/contact_angle_threshold_r4_wetted_area.csv`
- `validation/surface_tension_3d/contact_angle_timestep_r4_wetted_area.csv`
- `validation/surface_tension_3d/contact_angle_cost_r4_wetted_area.csv`
- `validation/surface_tension_3d/contact_angle_cost_r4_wetted_area_active.csv`
- `validation/surface_tension_3d/contact_angle_selected_matrix_r4_wetted_area.csv`
- `validation/surface_tension_3d/contact_angle_sensitivity_r4_wetted_area.csv`

R4-W is now eligible for a separate production-integration decision. It remains validation-only in
`compare_akinci/corrected_wetted_area_contact.jl`; no `src/` cache, API, export, or default changed,
and G3 remains open until that integration, D3/D5 cleanup, complete tests, and documentation pass.

## Production Integration Pre-registration

The production candidate is frozen as `WettedAreaContactAngle(theta)` and remains an explicit
`ColorfieldSurfaceNormal(contact_model=...)` choice. `ColorfieldSurfaceNormal()` continues to mean
no wetting model. Initial production support is deliberately limited to the configuration validated
by R4-W:

- three dimensions, `ContinuityDensity`, `WendlandC2Kernel{3}`, and `h/dx=1.4`;
- one fluid and one or more dummy-particle wall or rigid-body systems;
- explicit nonnegative `surface_measure` values on each contact boundary and
  `InitialCondition.normals` whose magnitudes give the particle-to-physical-wall offsets;
- one connected disk-like wetted patch per boundary system for the canonical edge correction;
- target angles strictly inside `(0,180)` degrees, with exact zero energy and force at 90 degrees.

The surface measure makes quadrature independent of global wall orientation and permits curved,
moving, and rigid surfaces when the caller supplies appropriate particle areas. It does not claim
support for disconnected contact patches on one boundary, changing topology, arbitrary kernels, or
unvalidated smoothing-length ratios.

Production cache ownership is also frozen. Fluid caches hold only the density conjugate, edge shift,
energy, and aggregate diagnostics. Each dummy boundary owns immutable surface measures plus transient
area weights and reaction forces. The colorfield/area pass computes the density conjugate once;
the density force is fused into the existing fluid-fluid RHS and the explicit derivative into the
existing fluid-boundary RHS. Fixed walls retain reaction diagnostics, while rigid bodies accumulate
the same reaction in `force_per_particle`.

Promotion gates are unchanged R4 physics plus:

1. constructor/configuration tests for dimensions, kernel, ratio, density formulation, quadrature,
   endpoints, Float32, multiple walls, moving orientation, and rigid reaction force/torque;
2. finite-difference energy gradients, exact 90-degree cancellation, and `1e-12` momentum/reaction
   residuals using only production caches and RHS paths;
3. active 60-degree median runtime overhead at most 20% over no contact, measured with the same three
   interleaved repetitions;
4. complete production-only static, perturbation, threshold, timestep, selected, and sensitivity
   replay in new `*_production.csv` files;
5. only after those pass, delete the unshipped rejected `GeometricContactAngle` and
   `ContactLineForce` implementations under D5, update all public prose/examples/metadata, and close
   G3 after the full verification suite.

## Production Integration Result

`WettedAreaContactAngle(theta)` now implements the frozen energy and complete derivative in
production. Boundary models accept explicit `surface_measure` quadrature, and configuration checks
enforce the validated dimension, kernel, smoothing ratio, density formulation, colors, wall offsets,
and connected-patch contract. The density term is fused into fluid-fluid WCSPH/EDAC interactions;
the explicit term and thread-local fixed-wall reaction are fused into fluid-boundary interactions.
Rigid reactions use the existing reverse rigid-fluid interaction and enter `force_per_particle`.

Production-only evidence passes static algebra `9/9` with maximum directional-gradient error
`3.59e-10`, the frozen extended perturbation `4/4`, threshold `5/5`, timestep `2/2`, selected matrix
`15/15`, and sensitivity `4/4` with `0.107` degree span. The exactly disabled 90-degree path has
0.4% median overhead; active 60-degree median overhead is 16.0%, passing the pre-registered 20%
gate. The rejected unshipped geometric and contact-line-force implementations were then deleted;
`contact_model=nothing` remains unchanged. Final verification passes 125/125 validation checks,
23063/23063 unit and Aqua checks, the documentation build, formatting, and relevant changed examples;
G3 is closed.

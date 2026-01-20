# Changelog

TrixiParticles.jl follows the interpretation of
[semantic versioning (semver)](https://julialang.github.io/Pkg.jl/dev/compatibility/#Version-specifier-format-1)
used in the Julia ecosystem. Notable changes will be documented in this file for human readability.

## Version 0.4.4

### Features

- Added `StateEquationAdaptiveCole` an adaptive sound speed version of the Cole state equation (#875)


## Version 0.4.3

### API Changes

- Added the keyword `coordinates_eltype` to `RectangularTank`,
  `RectangularShape` and `SphereShape`, which defaults to `Float64` (#956).

- Added keywords `element_type` and `coordinates_eltype` to `vtk2trixi` (#991)

### Features
- Added support for ASCII STL files with multiple patches as separate geometries,
  plus a function `extrude_geometry` and a union operation for `TrixiParticles.TriangleMesh` (#962).

- Added an `RCRWindkesselModel` open-boundary pressure model as a lumped-parameter
  description of vascular systems (#935, #1019, #992).

- Added support for FP64 coordinates when using FP32 GPU computations, fixing
  stability issues at high resolutions (#956).

- Added a new `LaguerreGaussKernel` smoothing kernel (#893).

- Added GPU and FP32 support for DEM (#979).

  
### Performance
- Improved GPU performance with shifting up to a factor of 10x (#974, #993).

- Significantly improved GPU performance of TLSPH (#1014).

## Version 0.4.2

### API Changes

- Keyword argument `n_clamped_particles` of the `TotalLagrangianSPHSystem`
  has been deprecated in favor of a new kwarg `clamped_particles`.

### Features

- Added `OscillatingMotion2D` to create an oscillating `PrescribedMotion` combining
  translation and rotation (#915).
- Added `BoundaryModelDynamicalPressureZhang` for `OpenBoundarySystem` (#900).
- Added `PrescribedMotion` to clamped particles in `TotalLagrangianSPHSystem` (#896).
- Added new boundary density calculator `PressureBoundaries` specifically for
  `ImplicitIncompressibleSPHSystem` (#946).
- Included wall velocity in interpolation (#941).
- 2D dam break validation now compares against the results from De Courcy et al. (#934).
- Improved performance of `TotalLagrangianSPHSystem` on GPUs (#968).

### Important Bugfixes

- Fixed transport velocity formulation with tensile instability control (#948).
- Fixed `TotalLagrangianSPHSystem` close to open boundaries (#954).
- `extrude_geometry` now doesn't adjust the particle spacing (#965).
- Reduced overhead of `UpdateCallback` when no update operations are performed (#973).

## Version 0.4.1

### Features

- Added an incompressible SPH solver `ImplicitIncompressibleSPHSystem` (#751).
- Added `SplitIntegrationCallback` to integrate the `TotalLagrangianSPHSystem`s
  in a `Semidiscretization` separately from the other systems (#794).

### Important Bugfixes

- Fixed `kernel_grad` threshold with single precision and small particle spacing (#913).
- Fixed error with `UpdateCallback` when `SaveSolutionCallback` comes first in the
  `CallbackSet` (#924).
- Fixed error when using shifting techniques without `UpdateCallback` (#925).
- Updated program flow diagram (#917).

## Version 0.4

### API Changes

- Renamed `BoundarySPHSystem` to `WallBoundarySystem` and the keyword argument
  `movement` to `prescribed_motion`.

- Renamed `OpenBoundarySPHSystem` to `OpenBoundarySystem`.

- Renamed `BoundaryMovement` to `PrescribedMotion`. The `movement_function` must now be
  a function of `(x, t)` returning the *new position* instead of an offset.
  For example, `movement_function(t) = SVector(t, 0.0)` now needs to be
  `movement_function(x, t) = x + SVector(t, 0.0)`.

- Renamed directory `solid` to `structure` in the examples file tree.
  VTK files for the `TotalLagrangianSPHSystem` are now also called `structure_*`.

- Renamed keyword argument `n_fixed_particles` of the `TotalLagrangianSPHSystem`
  to `n_clamped_particles`.

- API for `OpenBoundarySystem` and `BoundaryZone` changed.
  It is now possible to pass multiple `BoundaryZone`s to a single `OpenBoundarySystem`.
  Reference values are now assigned individually to each `BoundaryZone`. (#866)

- Rename keyword arguments `plane` and `plane_normal` for `BoundaryZone` to `boundary_face` and `face_normal` (#597).

- The argument of `TransportVelocityAdami` is now a keyword argument.
  `TransportVelocityAdami(1000.0)` now becomes
  `TransportVelocityAdami(background_pressure=1000.0)` (#884).

- Combined transport velocity formulation (TVF) and particle shifting technique (PST) into
  one unified framework.
  The keyword argument `transport_velocity` now changed to `shifting_technique`.
  The `ParticleShiftingCallback` has been removed. To use PST, use the `UpdateCallback`
  instead, and pass `shifting_technique=ParticleShiftingTechniqueSun2017()` to the system (#884).

- Renamed the keyword argument `tlsph` to `place_on_shell` for `ParticlePackingSystem`,
  `sample_boundary`, `extrude_geometry`, `RectangularShape`, and `SphereShape` (#814).

- Custom quantity functions passed to `SolutionSavingCallback` or `PostprocessCallback`
  that were not using the documented API but were functions of
  `(system, v_ode, u_ode, semi, t)` now need to be functions
  of `(system, dv_ode, du_ode, v_ode, u_ode, semi, t)` (#879).

- Renamed `each_moving_particle` to `each_integrated_particle`,
  `n_moving_particles` to `n_integrated_particles`
  and `active_particles` to `each_active_particle`.

### Features

- Added consistent particle shifting by Sun et al. (2019) as `ConsistentShiftingSun2019` (#888).


## Version 0.3.2
### Features

- **Open boundaries**:
  - Averaging of the inflow velocities has been added as an option. (#833)
  - New mirroring methods 0th order, 1st order and simple mirroring have been added. (#855)

- **TLSPH**:
  - The option to add artificial viscosity has been added to increase stability of FSI. (#869)


### Important Bugfixes

- Fix the coordinates used for TLSPH in Adami extrapolation (#853)
- Fix PST for small smoothing length factors (#834)
- The TVF model has been improved to integrate correctly with time stepping (#864)


## Version 0.3.1

### Features

- **Simplified SGS Viscosity Models**: Added ViscosityMorrisSGS and ViscosityAdamiSGS,
  which implement a simplified Smagorinsky-type sub-grid-scale viscosity. (#753)

- **Multithreaded Integration Array**: Introduced a new array type for CPU backends
  that enables multithreaded broadcasting, delivering speed-ups of up to 5× on systems
  with many threads when combined with thread pinning. (#722)

- **Tensile Instability Control (TIC)**: Implemented TIC to mitigate tensile
  instability artifacts in simulations. (#769)

- **DXF file format support**: Import complex geometries using the DXF file format. (#821)

- **Improved Plane interpolation**: Massively improved interpolation performance for planes (#763).

### GPU

 - Make PST GPU-compatible (#813).

 - Make open boundaries GPU-compatible (#773).

 - Make interpolation GPU-compatible (#812).

### Important Bugfixes

 - Fix validation setups (#801).

 - Calculate interpolated density instead of computed density when using interpolation (#808).

 - Fix allocations with Julia 1.10 (#809).

 - Fix Tafuni extrapolation for open boundaries (#829).

## Version 0.3

### API Changes

- Rescaled the Wendland kernels by a factor of 2 to be consistent with literature.
  This requires adjusting the previously used smoothing length for the Wendland Kernels
  by dividing them by 2 as well to obtain the same results (#775).

- API for custom quantities and functions in the `PostprocessCallback` changed (#755).

- The API for choosing parallelization backends changed. The keyword argument `data_type`
  in `semidiscretize` was removed and a keyword argument `parallelization_backend` was added
  to `Semidiscretization`. See [the docs on GPU support](@ref gpu_support) for more details.

### Features

- **Explicit Contact Models:** Added explicit contact models, `LinearContactModel` and `HertzContactModel`, to the DEM solver. (#756)

- **Particle Shifting Technique (PST) for Closed Systems:** Integrated the
  Particle Shifting Technique to enhance particle distribution, reduce clumping
  and prevent void regions due to tensile instability in closed system simulations. (#735)

- **Open Boundary Model:** Added an open boundary model based on Tafuni et al. (2018),
  utilizing mirroring and extrapolation to transfer fluid quantities to the buffer zones.
  This enhancement allows for more accurate handling of simulation boundaries in open systems,
  ensuring better consistency between the computed domain and its buffer areas. (#574)

- **Transport Velocity Formulation (TVF) for WCSPH Solver:** Added support for TVF
  to the WCSPH solver, improving the consistency and stability
  of weakly compressible SPH simulations. (#600)

### Refactoring

- **Variable Smoothing Length Structures:** Introduced new structures to support a variable
  smoothing length, providing enhanced flexibility in simulation configurations. (#736)

- **Flexible Parallelization Backend:** Improved the parallelization backend support,
  making it possible to switch the parallelization backend for single simulations. (#748)

- **Total Lagrangian SPH:** Added per-particle material parameters. (#740)

## Version 0.2.7

### Features

- Added the classic **Continuum Surface Force (CSF)** model based on Morris 2000 (#584), which computes
  surface tension as a **body force** proportional to curvature and directed along the interface normal.
  This method is efficient and accurate for capillary effects but does not explicitly conserve momentum.

- Added the classic **Continuum Surface Stress (CSS)** model based on Morris 2000 (#584), which is
  a **momentum-conserving** approach that formulates surface tension as the **divergence of a stress tensor**.
  However, it requires additional computation and stabilization to handle **high-density interfaces** and reduce numerical instabilities.

- Added `BoundaryZone` to allow for bidirectional flow (#623)

- Added the symplectic time integration scheme used in DualSPHysics (#716)

### Documentation

- Added documentation for time integration (#716)

### Testing

- Run CI tests on GPUs via Buildkite CI (#723)

### Bugs

- Fix GPU computations (#689)

## Version 0.2.6

### Features

- Support for surface tension was added to EDAC (#539)

### Refactored

- Surface normal calculation was moved from `surface_tension.jl` to `surface_normal_sph.jl` (#539)

## Version 0.2.5

### Features

- Add particle packing for 2D (.asc) and 3D (.stl) geometries (#529)

### Compatibility Changes

- Dropped support for Julia 1.9

## Version 0.2.4

### Features

- A method to prevent penetration of fast moving particles with solids was added (#498)
- Added the callback `SteadyStateReachedCallback` to detect convergence of static simulations (#601)
- Added ideal gas state equation (#607)
- Simulations can be run with `Float32` (#662)

### Documentation

- Documentation for GPU support was added (#660)
- A new user tutorial was added (#514)
- Several docs fixes (#663, #659, #637, #658, #664)

## Version 0.2.3

### Highlights

Transport Velocity Formulation (TVF) based on the work of Ramachandran et al. "Entropically damped artiﬁcial compressibility for SPH" (2019) was added.

## Version 0.2.2

### Highlights

Hotfix for threaded sampling of complex geometries.

## Version 0.2.1

### Highlights

Particle sampling of complex geometries from `.stl` and `.asc` files.

## Version 0.2.0

### Removed

Use of the internal neighborhood search has been removed and replaced with PointNeighbors.jl.

## Development Cycle 0.1

### Highlights

#### Discrete Element Method

A basic implementation of the discrete element method was added.

#### Surface Tension and Adhesion Model

A surface tension and adhesion model based on the work by Akinci et al., "Versatile Surface Tension and Adhesion for SPH Fluids" (2013) was added to WCSPH.

#### Support for Open Boundaries

Open boundaries using the method of characteristics based on the work of Lastiwka et al., "Permeable and non-reflecting boundary conditions in SPH" (2009) were added for WCSPH and EDAC.

## Pre Initial Release (v0.1.0)

This section summarizes the initial features that TrixiParticles.jl was released with.

### Highlights
#### EDAC

An implementation of EDAC (Entropically Damped Artificial Compressibility) was added,
which allows for more stable simulations compared to basic WCSPH and reduces spurious pressure oscillations.

#### WCSPH

An implementation of WCSPH (Weakly Compressible Smoothed Particle Hydrodynamics), which is the classical SPH approach.

Features:

- Correction schemes (Shepard (0. Order) ... MixedKernelGradient (1. Order))
- Density reinitialization
- Kernel summation and Continuity equation density formulations
- Flexible boundary conditions e.g. dummy particles with Adami pressure extrapolation, pressure zeroing, pressure mirroring...
- Moving boundaries
- Density diffusion based on the models by Molteni & Colagrossi (2009), Ferrari et al. (2009) and Antuono et al. (2010).


#### TLSPH

An implementation of TLSPH (Total Lagrangian Smoothed Particle Hydrodynamics) for solid bodies enabling FSI (Fluid Structure Interactions).



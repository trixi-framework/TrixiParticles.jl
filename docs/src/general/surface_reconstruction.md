# [Surface Reconstruction](@id surface_reconstruction)

For three-dimensional simulations, a continuous free surface can be reconstructed from
the fluid particles with [`SurfaceReconstruction`](@ref), either directly from particle
snapshots, systems, solutions, or initial conditions with [`reconstruct_surface`](@ref),
or during a simulation with [`SurfaceReconstructionCallback`](@ref), which shares the
same engine. See the [postprocessing example](@ref examples) for usage
(`examples/postprocessing/surface_reconstruction_3d.jl`).

This page documents the reconstruction algorithm and its configuration. The pipeline
below is the configuration validated and used in production; only these settings are
part of the package.

## Pipeline

1. **Particle volumes.** Each fluid particle contributes its volume ``V_i = m_i / \rho_i``
   from the constant particle mass and the current density. Inactive buffer particles
   are excluded.
2. **Volume deposition.** The volumes are deposited on a uniform voxel grid with
   trilinear cloud-in-cell (CIC) interpolation, whose weights sum to one in real
   arithmetic [Birdsall1969](@cite). The implementation accumulates a `Float32` field,
   so deposited volume is conserved up to roundoff. Particles whose complete stencil
   does not fit are excluded from both deposition and the correction target. The default
    grid spacing is half the particle spacing (``h/2``).
3. **Gaussian filtering.** A separable Gaussian filter with width ``0.9h`` by default
   smooths the volume field and suppresses grid-scale noise.
4. **Domain and boundary constraints.** The field is intersected with the implicit
   level sets of the tank interior and of closed boundary meshes (e.g. moving structures),
   whose signed distances are evaluated with a triangle BVH using angle-weighted
   pseudonormals [Ericson2004](@cite). The production tank is open at the top; custom
   clipping boxes are configured with `min_corner`/`max_corner`/`open_faces`.
5. **Isovalue correction.** The marching-cubes isovalue is adjusted with a safeguarded
   regula falsi iteration until the enclosed mesh volume matches the retained particle
   volume (default tolerance ``0.1\%``, at most 8 refinement iterations in addition to
   the initial and bracketing evaluations), following the
   volume-control idea of [Zhu2005](@cite). The correction is warm-started from the
   previous frame for time series.
6. **Contouring.** The ``0.5`` level set (configurable) is extracted with marching cubes
   [Lorensen1987](@cite), using the classic lookup table from MarchingCubes.jl. This is
   its `march_legacy` algorithm, not the topology-aware algorithm of [Lewiner2003](@cite).
7. **Mesh cleanup and analysis.** Near-duplicate vertices are merged, degenerate
   triangles collapsed, and the mesh is decomposed into liquid domains, nested shells,
   and cavity regions. [`write_ply`](@ref), [`trixi2vtk`](@ref), and
   `TriangleMesh` conversion orient outer shells outward and cavities inward in the
   simulation coordinate system. The raw `SurfaceMesh` retains contour ordering.

Before returning, the pipeline checks the final mesh's volume tolerance and rejects
boundary edges, nonmanifold edges, and degenerate triangles. These edge checks are not
a proof of global manifoldness or absence of self-intersections. The returned
[`SurfaceReconstructionStatistics`](@ref) records
the effective isovalue, volumes, component counts, and per-stage timings for auditing.

## Callback output

[`SurfaceReconstructionCallback`](@ref) writes, per reconstruction event:

- one VTK PolyData file per fluid system (`surface_fluid_1_<iter>.vtp`) with per-vertex
  normals, collected in a ParaView time series (`surface_fluid_1.pvd`); `formats=(:vtp, :ply)`
  adds PLY files for Blender;
- optionally SPH quantities interpolated onto the surface vertices
  (`interpolated_quantities=(:velocity, :pressure, :density)`), e.g. to color the surface
  by velocity. The interpolation uses [`interpolate_points`](@ref) with Shepard
  normalization over fluid particles only, so values are convex combinations of particle
  values; vertices without fluid neighbors get `NaN`;
- optionally the tracked surfaces of boundary and structure systems passed as
  `boundaries` (`write_boundaries=true`), e.g. deforming structures for rendering.

In addition, `surface_statistics.csv` and `surface_statistics.json` hold the time series
of the reconstruction statistics of every fluid system (enclosed volume, particle volume,
relative volume error, effective isovalue, region and topology-defect counts, correction
evaluations, run time) in the format of the [`PostprocessCallback`](@ref), which makes
volume conservation and reconstruction quality auditable over a whole simulation. The
metadata file `meta_surface.json` records the simulation, the systems, and the full
reconstruction configuration.

## Sparse-component fallback

Isolated droplets and thin splashes below the grid resolution do not form a level set.
With `sparse_component_fallback=true`, compact particle components that the primary
surface misses are filled with volume-equivalent sphere meshes (see the
[surface reconstruction example](@ref examples)). This is a *visualization prior*, not
a physical model: it preserves the total volume for rendering but invents geometry.
It is enabled by default in the accompanying example and was used in production for
this reason. The current fallback requires a resolvable primary liquid region and
compact unresolved components of at most 64 particles. It rejects intersecting fallback
spheres and spheres that cross sampled solid constraints; it is not a general
reconstruction method for an entirely unresolved spray.

## Determinism

For identical particle arrays on the same floating-point platform, reconstruction uses
disjoint grid writes and fixed-order reductions to make threaded results independent of
thread scheduling and thread count. Large-mesh volume and area sums use 64 fixed
buckets. `SerialBackend()` preserves the original serial sum order instead. Their
observed relative difference on the benchmark meshes is about `1e-12`; this is a
measurement, not a universal error bound. Cancellation can increase relative error, and
changes in computed volume can change isovalue selection or topology near a threshold.
Serial and threaded results are therefore not promised to be bitwise identical.

Fully order-preserving parallel work is preferred where practical; floating-point
non-associativity does not prohibit parallel computation of independent summands.
The GPU simulation and interpolation have their own neighbor-order behavior; the
reconstruction guarantee does not imply bitwise reproducibility of the entire simulation.

## Numerical contract and limitations

- The separable Gaussian is a tensor-product convolution with a truncated, normalized
  kernel. Zero padding can lose mass near the edge of the finite grid. Volume correction
  targets particle volume, not the filtered-field integral.
- On grid nodes, taking the minimum of the fluid and constraint fields represents their
  intersection. Marching cubes linearly interpolates these samples: it does not perform
  exact geometric clipping against the original boundary triangles. Boundary clearance
  is therefore resolution-dependent.
- Shell volumes use the oriented tetrahedron formula and nesting parity. This formula
  assumes closed, consistently oriented, non-self-intersecting shells with a valid
  containment hierarchy. BVH signed distances likewise require an embedded oriented
  boundary. Open or inconsistently oriented boundary edges are rejected; a general
  triangle-intersection test is not performed.
- The classic marching-cubes table does not guarantee the topology of the underlying
  trilinear level set in ambiguous cells. The correction can fail to bracket or converge
  within its iteration budget; in that case it raises an error. A successful return
  certifies the computed final volume tolerance, not a unique or physically exact surface.
  With its edge-index cache cleared, Lewiner's case table correctly connects a
  manufactured saddle that classic marching cubes splits at coarse resolution. But
  MarchingCubes.jl 0.1.11 uses an absolute floating-point threshold in its ambiguity
  tests: low-amplitude closed fields can produce nonmanifold edges. Switching the
  default needs a corrected, validated implementation and a new volume-correction and
  performance assessment.
- Mesh vertices use `Float32` world coordinates. Very large offsets relative to grid
  spacing can lose geometric resolution. Cleanup uses fixed absolute length (`1e-6`),
  area (`1e-14`), and near-zero-volume (`eps(Float64)`) thresholds in simulation units;
  arbitrary rescaling is not guaranteed to preserve the mesh. Choose units and grid
  spacing that resolve the features of interest.

## Performance

The reconstruction runs multithreaded on the CPU with the CPU backend of the simulation;
GPU simulations are reconstructed from a CPU copy of the particle data. The voxel grid is
dense and needs about 16 bytes per voxel, while most grid passes on later frames are
restricted to the occupied region. For example, a 2 m × 4 m × 1 m tank with a particle
spacing of 6.9 mm has a 599 × 1180 × 309 grid at the default voxel size (3.5 GB). On
one 24-thread workstation, a synthetic frame with 1.1 million particles takes about
0.2–0.3 seconds after workspace allocation; the runtime depends on the fluid geometry,
correction evaluations, and hardware. On GPUs (Float32), only state vectors and systems
transfer to the CPU each event — the neighborhood-search handler transfer is skipped
unless `interpolated_quantities` are requested — and VTK output defaults to compression
level 1, which writes almost as small as level 6 at a fifth of the cost.

## Related methods

Anisotropic kernel reconstruction [Yu2010](@cite) adapts the smoothing shape to the
local particle distribution and resolves thin films better, but was evaluated against
the volume-CIC default during development and is not part of the package.

## API

```@autodocs
Modules = [TrixiParticles]
Pages = ["surface_reconstruction/surface_reconstruction.jl",
         "surface_reconstruction/mesh.jl",
         "surface_reconstruction/grid.jl",
         "surface_reconstruction/constraints.jl",
         "surface_reconstruction/boundaries.jl",
         "surface_reconstruction/io.jl"]
```

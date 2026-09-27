# [Surface Reconstruction](@id surface_reconstruction)

For two- and three-dimensional simulations, a continuous free surface can be reconstructed from
the fluid particles with [`SurfaceReconstruction`](@ref), either directly from particle
snapshots, systems, solutions, or initial conditions with [`reconstruct_surface`](@ref),
or during a simulation with [`SurfaceReconstructionCallback`](@ref), which shares the
same engine. See the [postprocessing example](@ref examples) for usage
(`examples/postprocessing/surface_reconstruction_2d.jl` and
`examples/postprocessing/surface_reconstruction_3d.jl`).

This page documents the reconstruction algorithm and its configuration. Both dimensions
use particle-measure deposition, Gaussian smoothing, and isovalue correction; planar
contours use area and the production 3D surface pipeline uses volume.

```@eval
using CairoMakie
using LinearAlgebra: cross, dot, normalize
using TrixiParticles

const GB = CairoMakie.GeometryBasics

# Both figures use the same deterministic wave-shaped particle snapshot. The first
# panel displays the actual particles; the mesh, grid fields, and contour all come
# from `reconstruct_surface!`, rather than from an independently drawn outline.
function wave_particles(spacing)
    positions = SVector{3, Float64}[]
    for z_index in 0:(floor(Int, 1.0 / spacing) - 1),
        x_index in 0:(floor(Int, 2.0 / spacing) - 1)
        x = (x_index + 0.5) * spacing
        z = (z_index + 0.5) * spacing
        height = 0.3 + 0.1 * sin(2pi * x)
        for y_index in 0:floor(Int, height / spacing)
            push!(positions, SVector(x, (y_index + 0.5) * spacing, z))
        end
    end
    return reduce(hcat, positions), fill(spacing^3, length(positions))
end

function plot_coordinates(vertex)
    # Axis3's vertical coordinate is z; simulation gravity points along -y.
    return GB.Point3f(vertex[1], vertex[3], vertex[2])
end

function plot_surface(mesh)
    # Duplicate face vertices so each triangle can have its own color from its
    # physical normal, without adding a smoothing artifact to the illustration.
    positions = GB.Point3f[]
    faces = GB.TriangleFace{Int32}[]
    colors = RGBf[]
    light = normalize(SVector(0.4f0, 0.3f0, 0.85f0))
    for face in mesh.faces
        a, b, c = (plot_coordinates(mesh.vertices[index]) for index in face)
        normal = normalize(cross(b - a, c - a))
        shade = 0.45f0 + 0.5f0 * abs(dot(normal, light))
        color = RGBf(0.03f0, 0.25f0 + 0.32f0 * shade, 0.48f0 + 0.37f0 * shade)
        start = length(positions)
        append!(positions, (a, b, c))
        push!(faces, GB.TriangleFace(start + 1, start + 2, start + 3))
        append!(colors, (color, color, color))
    end
    return positions, faces, colors
end

function main()
    spacing = 0.07
    points, volumes = wave_particles(spacing)
    rec = SurfaceReconstruction(; particle_spacing=spacing, tank_size=(2.0, 0.65, 1.0))
    surface, stats = reconstruct_surface!(rec, points, volumes)
    workspace = rec.cache.workspace[]
    @assert stats.n_boundary_edges == 0

    # Identical cameras show the input particles and reconstructed mesh side by side.
    figure = Figure(; size=(1200, 510), fontsize=17, backgroundcolor=:white)
    camera = (; elevation=0.36, azimuth=1.2, aspect=(2, 1, 0.85))
    ax_particles = Axis3(figure[1, 1]; title="Fluid particles",
                         xlabel="x (m)", ylabel="z (m)", zlabel="height (m)", camera...)
    ax_surface = Axis3(figure[1, 2]; title="Volume-corrected surface",
                       xlabel="x (m)", ylabel="z (m)", zlabel="height (m)", camera...)
    scatter!(ax_particles, [plot_coordinates(view(points, :, p))
                            for p in axes(points, 2)];
             color=(:steelblue, 0.67), markersize=2.6)
    vertices, faces, colors = plot_surface(surface)
    mesh!(ax_surface, vertices, faces; color=colors, shading=false)
    for ax in (ax_particles, ax_surface)
        limits!(ax, (0, 2), (0, 1), (0, 0.55))
    end
    colgap!(figure.layout, 50)
    save("surface_reconstruction_wave.png", figure; px_per_unit=1.5)

    # The same z cross-section of the particles, raw CIC grid, and filtered grid.
    # The white contour is the corrected isovalue subject to the actual tank constraint.
    field = workspace.field
    raw = zeros(Float32, size(field))
    TrixiParticles.deposit_volume_cic!(raw, points, volumes, workspace.origin,
                                        rec.voxel_size)
    z_index = round(Int, (0.5 - workspace.origin[3]) / rec.voxel_size) + 1
    x_nodes = [workspace.origin[1] + (i - 1) * rec.voxel_size
               for i in axes(field, 1)]
    y_nodes = [workspace.origin[2] + (j - 1) * rec.voxel_size
               for j in axes(field, 2)]
    scalar = min.(field[:, :, z_index] .- Float32(stats.effective_isovalue),
                  workspace.constraint[:, :, z_index])
    cross_section = Figure(; size=(1300, 390), fontsize=17, backgroundcolor=:white)
    panels = [Axis(cross_section[1, column];
                   title=title, xlabel="x (m)", ylabel="height (m)",
                   limits=((0, 2), (0, 0.58)))
              for (column, title) in enumerate(("Particles", "CIC deposition",
                                                 "Filtered field + contour"))]
    indices = [p for p in axes(points, 2) if abs(points[3, p] - 0.5) < spacing / 2]
    scatter!(panels[1], points[1, indices], points[2, indices];
             color=:steelblue, markersize=5)
    for (ax, data) in ((panels[2], raw[:, :, z_index]),
                       (panels[3], field[:, :, z_index]))
        heatmap!(ax, x_nodes, y_nodes, data; colormap=:viridis,
                 colorrange=(0, 1.1))
    end
    contour!(panels[3], x_nodes, y_nodes, scalar; levels=[0], color=:white,
             linewidth=3)
    save("surface_reconstruction_pipeline.png", cross_section; px_per_unit=1.5)
    return nothing
end

main()
```

![Wave-shaped fluid particles beside their reconstructed triangle surface](surface_reconstruction_wave.png)

The same sinusoidal particle snapshot (spacing ``h=0.07`` m) and its volume-corrected
surface. The vertical axis is enlarged for clarity in this shallow tank. The wave is
an illustration, not a simulated result.

## Planar contours (2D)

In 2D, particle ``m_i/\rho_i`` is **area**. The same API deposits it with bilinear CIC,
filters in two directions, and extracts **line segments** using marching squares. A
bilinear asymptotic decider resolves alternating-sign saddle cells; products are
evaluated in `Float64` without an absolute ambiguity tolerance. An exactly zero saddle
determinant is a topological tie and uses a fixed pairing. The correction targets liquid
area. Samples exactly at the contour level use a small positive perturbation relative to
their neighbors to avoid coincident edge vertices. The loop analysis computes liquid
area, with nested holes subtracted and islands added. Outer loops are counterclockwise,
hole loops clockwise, and normals point out of liquid in the simulation plane.

At an exact saddle tie, either connection is a valid perturbation of the degenerate
bilinear zero set; the fixed pairing is a convention, not a uniquely correct topology.
Exact-zero nodes are perturbed only while extracting a contour: an evaluation with **no
strictly positive nodes** is treated as empty. This avoids inventing area from an
all-zero plateau, while fields with positive nodes may still contain zero-node contours.

```julia
fluid = RectangularShape(0.05, (16, 8), (0.0, 0.0); density=1000.0)
contour, stats = reconstruct_surface(fluid; tank_size=(1.2, 0.8))
stats["area"], stats["perimeter"]

# For repeated adaptive frames, select the dimension explicitly.
reconstruction = SurfaceReconstruction(; particle_spacing=0.05, ndims=2)
contour, stats = reconstruct_surface!(reconstruction, fluid.coordinates,
                                      fluid.mass ./ fluid.density)
trixi2vtk(contour; output_directory="out", filename="contour")
```

One-shot calls infer dimension from particle arrays, systems, or initial conditions.
For a reusable `SurfaceReconstruction`, `tank_size` or domain corners infer dimension;
without either, use `ndims=2` (the compatibility default is 3). Planar `open_faces` uses
`(-x, +x, -y, +y)` order. A two-component `tank_size` opens the `+y` face.

```@eval
using CairoMakie
using TrixiParticles

fluid = RectangularShape(0.05, (16, 8), (0.0, 0.0); density=1000.0)
contour, stats = reconstruct_surface(fluid; tank_size=(1.2, 0.8))
figure = Figure(; size=(760, 400), fontsize=17)
ax = Axis(figure[1, 1]; xlabel="x (m)", ylabel="y (m)", aspect=DataAspect(),
          title="Area-corrected planar contour")
scatter!(ax, fluid.coordinates[1, :], fluid.coordinates[2, :];
         color=:steelblue, markersize=6, label="Particles")
segments = [Point2f(contour.vertices[index]...) for edge in contour.faces for index in edge]
linesegments!(ax, segments; color=:darkorange, linewidth=3, label="Reconstructed contour")
Legend(figure[2, 1], ax; orientation=:horizontal, framevisible=false)
save("surface_reconstruction_2d.png", figure; px_per_unit=1.5)
nothing
```

![Planar particles and their closed area-corrected contour](surface_reconstruction_2d.png)

Planar meshes use `SurfaceMesh{Float64, Int32, 2}` with two-index `faces`. Float64
coordinates retain intersections close to grid nodes. Closed `Polygon{2}` geometries
can be converted with `BoundaryMesh(polygon)`; complete 2D particle lattices can be
tracked through `lattice_surface_topology` just like their 3D counterparts. Polygon
boundaries and combined fallback contours are checked for nonadjacent segment crossings.
No thickness or filled triangulation is added to line output.

The existing `volume_tolerance_percent` controls relative **area** error in 2D.
The typed statistics fields `volume`, `particle_volume`, and `surface_area` denote the
dimension's particle measure and boundary measure; explicit 2D aliases are
`stats["area"]`, `stats["particle_area"]`, and `stats["perimeter"]`. The callback writes
`area`, `particle_area`, `relative_area_error`, `perimeter`, and `n_segments` time series.
The current `cavity_volume` statistic denotes **gross** odd-depth hole area in 2D,
including liquid islands inside those holes. In 3D it denotes
**net** void volume after subtracting directly enclosed islands. This difference does
not affect the reconstructed liquid area/volume or its correction target. For net 2D
void area, subtract the areas of even-depth shells at depth two or greater from
`cavity_volume`, using `shell_volumes` and `shell_nesting_depths`.

As in 3D, topology transitions can prevent the correction from meeting a prescribed
tolerance, in which case reconstruction raises an error.

## 3D pipeline

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

![Cross-section from particles through CIC deposition and filtering to the constrained contour](surface_reconstruction_pipeline.png)

At the same cross-section, CIC deposits particle volume on grid nodes, Gaussian filtering
smooths the field, and the white line is the final constraint-clipped, corrected contour.

Before returning, the pipeline checks the final mesh's volume tolerance and rejects
boundary edges, nonmanifold edges, and degenerate triangles. These edge checks are not
a proof of global manifoldness or absence of self-intersections. The returned
[`SurfaceReconstructionStatistics`](@ref) records
the effective isovalue, volumes, component counts, and per-stage timings for auditing.

## Boundary placement and clearance

`BoundaryMesh` formed from `lattice_surface_topology` connects the **outermost particle
centers**, not a separately specified physical wall. For a uniform rectangular shape
whose centers begin half a particle spacing inside the physical face, the mesh lies
about ``h/2`` inside that face. For example, with ``h=0.05`` m, a physical face at
``x=0.25`` m has its lattice boundary at ``x=0.275`` m. The default
`boundary_clearance=0` clips at the lattice mesh; it does not infer this offset.

If the intended surface is the physical wall, set a clearance appropriate to the
boundary discretization (often ``h/2`` for that rectangular lattice), or pass geometry
that represents the physical wall directly. The tank-size constraint itself is placed
at the specified tank coordinates. This choice changes where volume is redistributed
by area/volume correction, so it should be recorded with the reconstruction settings.

## Callback output

[`SurfaceReconstructionCallback`](@ref) writes, per reconstruction event:

- one VTK PolyData file per fluid system (`surface_fluid_1_<iter>.vtp`) with per-vertex
  normals, collected in a ParaView time series (`surface_fluid_1.pvd`); `formats=(:vtp, :ply)`
  adds PLY files. In 2D, these contain VTK line cells or PLY edge elements in the `z=0`
  plane, with in-plane normals;
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
In 2D the same option uses area-equivalent regular polygons approximating circles;
their polygonal area, rather than ``\pi r^2``, matches the particle target. The fallback
requires compact components and rejects circles that could cross solid constraints.

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
- 3D mesh vertices use `Float32` world coordinates. Very large offsets relative to grid
  spacing can lose geometric resolution. The 3D cleanup uses fixed absolute length (`1e-6`),
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

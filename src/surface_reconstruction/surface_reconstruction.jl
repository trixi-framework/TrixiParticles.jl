# Public API for 2D and 3D free-surface reconstruction.
#
# The included files provide the numerical kernel: trilinear volume-CIC
# deposition, a separable Gaussian filter, implicit tank and boundary level-set
# constraints, safeguarded isovalue volume correction, marching-cubes contouring, and
# topology-preserving mesh cleanup in 3D. The planar path uses bilinear deposition,
# marching squares, and nested-loop area correction. Research field methods and
# correction modes are intentionally not part of the package.
# The wrappers below expose the kernel as a reusable, stateful
# `SurfaceReconstruction` object that owns the reconstruction grid and workspace, supports
# warm-started repeated invocations, and integrates with live simulations through
# `SurfaceReconstructionCallback` (see `src/callbacks/surface_reconstruction.jl`).
include("configuration.jl")
include("deposition.jl")
include("boundaries.jl")
include("sparse_fallback.jl")
include("filtering.jl")
include("constraints.jl")
include("mesh.jl")
include("io.jl")
include("grid.jl")
include("mesh_2d.jl")
include("contouring.jl")
include("pipeline.jl")
include("pipeline_2d.jl")

# 3D vertices are Float32, planar vertices Float64, and indices Int32. Writers and
# TriangleMesh conversion orient shells by nesting; 3D contours retain MC ordering.

# Liquid analysis of the last returned mesh, paired with that mesh object (or `nothing`)
const LastMeshAnalysis = Union{Nothing,
                               Tuple{SurfaceMesh{Float32, Int32, 3}, MeshLiquidAnalysis},
                               Tuple{SurfaceMesh{Float64, Int32, 2}, ContourAnalysis}}

const SurfaceWorkspace = Union{Nothing, ReconstructionWorkspace, ReconstructionWorkspace2D}

"""
    SurfaceReconstructionCache

Mutable invocation state of a [`SurfaceReconstruction`](@ref): the reusable grid
workspace and the previous effective isovalue for warm starts. Configuration itself is
immutable; only the cache is mutated by [`reconstruct_surface!`](@ref).
"""
struct SurfaceReconstructionCache
    workspace::Ref{SurfaceWorkspace}
    previous_isovalue::Ref{Float64}
    # Liquid analysis of the mesh returned by the last `reconstruct_surface!`, paired
    # with that mesh object (or `nothing` when the final mesh differs from the analyzed
    # one, e.g. after a sparse-fallback combine). The callback passes it to the surface
    # writers so they skip their own liquid analysis; it must only be used with the
    # stored mesh object.
    last_mesh_analysis::Ref{LastMeshAnalysis}
end

"""
    SurfaceReconstructionStatistics

Typed summary of a reconstructed surface. All lengths, areas, and volumes are in
simulation units. Fields:

- `effective_isovalue`: Contour level after the volume correction.
- `volume`: Enclosed volume in 3D, enclosed liquid area in 2D.
- `surface_area`: Surface area in 3D, contour perimeter in 2D.
- `particle_volume`: Total particle volume Σᵢ mᵢ/ρᵢ targeted by the
  volume correction.
- `n_connected_regions`, `n_cavity_regions`: Number of liquid regions and of enclosed
  cavities.
- `n_boundary_edges`, `n_nonmanifold_edges`, `n_degenerate_triangles`: Topology defects.
  Zero counts do not alone prove the mesh has no self-intersections or vertex-link defects.
- `n_vertices_inside_boundaries`: Surface vertices inside boundary meshes.
- `grid_dimensions`: Voxel dimensions of the reconstruction grid.
- `n_correction_evaluations`: Contour evaluations of the volume correction.
- `details`: The complete record (per-stage `timings` in seconds, correction evaluations,
  shell volumes, sparse-fallback records), also returned by [`statistics_dict`](@ref)
  for JSON export. `stats["key"]` works for every field and every key of the record.

In 2D, `stats["area"]`, `stats["particle_area"]`, and `stats["perimeter"]` are explicit
aliases. The `volume` fields retain the SPH convention of particle measure in the
simulation dimension; no artificial thickness is introduced.
"""
struct SurfaceReconstructionStatistics
    effective_isovalue::Float64
    volume::Float64
    surface_area::Float64
    particle_volume::Float64
    n_connected_regions::Int
    n_cavity_regions::Int
    n_boundary_edges::Int
    n_nonmanifold_edges::Int
    n_degenerate_triangles::Int
    n_vertices_inside_boundaries::Int
    grid_dimensions::Vector{Int}
    n_correction_evaluations::Int
    details::Dict{String, Any}
end

function SurfaceReconstructionStatistics(details::Dict{String, Any})
    return SurfaceReconstructionStatistics(details["effective_isovalue"],
                                           details["volume"],
                                           details["surface_area"],
                                           details["particle_volume"],
                                           details["n_connected_regions"],
                                           details["n_cavity_regions"],
                                           details["n_boundary_edges"],
                                           details["n_nonmanifold_edges"],
                                           details["n_degenerate_triangles"],
                                           details["n_vertices_inside_boundaries"],
                                           details["grid_dimensions"],
                                           length(details["isovalue_correction_evaluations"]),
                                           details)
end

function Base.getindex(stats::SurfaceReconstructionStatistics, key::String)
    _statistics_value(stats, key)
end

function _statistics_value(stats, key)
    symbol = Symbol(key)
    if symbol in fieldnames(SurfaceReconstructionStatistics) && symbol !== :details
        return getfield(stats, symbol)
    end

    return stats.details[key]
end

function Base.haskey(stats::SurfaceReconstructionStatistics, key::String)
    haskey(stats.details, key)
end
Base.keys(stats::SurfaceReconstructionStatistics) = keys(stats.details)

"""
    statistics_dict(stats)

Return the complete reconstruction record (per-stage timings, correction evaluations,
shell volumes, sparse-fallback records) as a `Dict{String, Any}`, e.g. for JSON export.
"""
statistics_dict(stats::SurfaceReconstructionStatistics) = stats.details

function Base.show(io::IO, stats::SurfaceReconstructionStatistics)
    @nospecialize stats # reduce precompilation time

    print(io, "SurfaceReconstructionStatistics(volume=", stats.volume,
          ", isovalue=", stats.effective_isovalue, ")")
end

function Base.show(io::IO, ::MIME"text/plain", stats::SurfaceReconstructionStatistics)
    @nospecialize stats # reduce precompilation time

    if get(io, :compact, false)
        show(io, stats)
    else
        setup = [
            "effective isovalue" => stats.effective_isovalue,
            (length(stats.grid_dimensions) == 2 ? "area" : "volume") => stats.volume,
            (length(stats.grid_dimensions) == 2 ? "perimeter" : "surface area") => stats.surface_area,
            "connected regions" => stats.n_connected_regions,
            "cavity regions" => stats.n_cavity_regions,
            "boundary edges" => stats.n_boundary_edges,
            "nonmanifold edges" => stats.n_nonmanifold_edges,
            "grid dimensions" => stats.grid_dimensions,
            "correction evaluations" => stats.n_correction_evaluations
        ]
        summary_box(io, "SurfaceReconstructionStatistics", setup)
    end
end

"""
    SurfaceReconstruction

Immutable configuration for 2D or 3D free-surface reconstruction from SPH particles.
Mutable invocation state (reusable workspace, warm-start isovalue) lives in
[`SurfaceReconstructionCache`](@ref). See the [`SurfaceReconstruction`](@ref)
constructor for keyword documentation.
"""
struct SurfaceReconstruction{NDIMS, NFACES}
    options::SurfaceReconstructionOptions
    particle_spacing::Float64
    voxel_size::Float64
    sigma_voxels::Float64
    min_corner::Union{Nothing, SVector{NDIMS, Float64}}
    max_corner::Union{Nothing, SVector{NDIMS, Float64}}
    open_faces::NTuple{NFACES, Bool}
    boundary_clearance::Float64
    grid_padding::Float64
    parallelization_backend::Union{Nothing, PointNeighbors.AbstractThreadingBackend}
    cache::SurfaceReconstructionCache

    function SurfaceReconstruction{N, F}(options, particle_spacing, voxel_size,
                                         sigma_voxels,
                                         min_corner, max_corner, open_faces,
                                         boundary_clearance,
                                         grid_padding, parallelization_backend,
                                         cache) where {N, F}
        N in (2, 3) && F == 2N || throw(ArgumentError("invalid reconstruction dimension"))
        return new{N, F}(options, particle_spacing, voxel_size, sigma_voxels,
                         min_corner, max_corner, open_faces, boundary_clearance,
                         grid_padding, parallelization_backend, cache)
    end
end

# Derive the dimension from the face flags even for adaptive grids, whose two corners
# are `nothing`. This also supplies the constructor used by Accessors for cache updates.
function SurfaceReconstruction(options, particle_spacing, voxel_size, sigma_voxels,
                               min_corner, max_corner, open_faces::NTuple{F, Bool},
                               boundary_clearance, grid_padding, parallelization_backend,
                               cache) where {F}
    return SurfaceReconstruction{F ÷ 2, F}(options, particle_spacing, voxel_size,
                                           sigma_voxels,
                                           min_corner, max_corner, open_faces,
                                           boundary_clearance,
                                           grid_padding, parallelization_backend, cache)
end

Base.ndims(::SurfaceReconstruction{N}) where {N} = N

"""
    SurfaceReconstruction(; particle_spacing, kwargs...)

Reusable configuration and state for 2D or 3D free-surface reconstruction from SPH particles.

The reconstruction deposits per-particle measures `V_i = m_i / rho_i` with bilinear (2D)
or trilinear (3D) cloud-in-cell interpolation, filters the field with a separable Gaussian, intersects it
with the tank interior and boundary signed-distance fields, and corrects the isovalue per
frame until the enclosed mesh volume matches the retained particle volume within the
configured tolerance (or throws an error if correction fails).

A `SurfaceReconstruction` owns the reconstruction grid and the internal workspace, so
repeated invocations of [`reconstruct_surface!`](@ref) reuse allocated buffers and
warm-start the isovalue correction from the previous frame. Pass `tank_size` to pin the
reconstruction grid, which guarantees workspace reuse across frames. Without `tank_size`,
the grid is derived from the particle bounds of each invocation and the workspace is
rebuilt whenever the grid dimensions change.

# Keywords
- `particle_spacing`:            Particle spacing of the fluid discretization (required).
- `ndims=nothing`:               Dimension, 2 or 3. Inferred from `tank_size` or domain
                                 corners when supplied; otherwise defaults to 3. Use
                                 `ndims=2` for an adaptive planar reconstruction. One-shot
                                 particle/system entry points infer the dimension from input.
- `voxel_size=nothing`:          Reconstruction grid spacing
                                 (default: `particle_spacing / 2`, the production setting).
- `gaussian_sigma=nothing`:    Gaussian smoothing width in length units
                                 (default: `0.9 * particle_spacing`, the production setting).
- `gaussian_sigma_voxels=nothing`: Gaussian smoothing width in voxels. Mutually exclusive
                                 with `gaussian_sigma`.
- `tank_size=nothing`:           Size `(x, y)` or `(x, y, z)` of the fluid domain, equivalent to
                                 a zero `min_corner`, `max_corner=tank_size` with only
                                 the `+y` face open (the production tank convention).
                                 Used to pin the reconstruction grid and to constrain
                                 the reconstructed surface to the tank interior. Cannot
                                 be combined with `min_corner`/`max_corner`/`open_faces`.
- `min_corner=nothing`:          Lower interior corner of the clipping domain. With
                                 `max_corner`, pins the reconstruction grid instead of
                                 `tank_size`. With `nothing`, the grid is adaptive and
                                 unconstrained.
- `max_corner=nothing`:          Upper interior corner of the clipping domain (requires
                                 `min_corner`).
- `open_faces=nothing`:          Four (2D) or six (3D) flags selecting faces that are not
                                 walls, in `(-x, +x, -y, +y[, -z, +z])` order. Defaults to all
                                 closed; only valid with `min_corner`/`max_corner`.
 - `boundary_clearance=0.0`:      Extra clearance from the boundary mesh, not necessarily
                                 from the physical wall. A lattice-derived mesh passes
                                 through the outer particle centers; for a uniform
                                 rectangular lattice these lie about half a particle
                                 spacing inside its physical surface. Specify a suitable
                                 clearance (often `particle_spacing/2`) if the latter is
                                 the intended clipping surface. Meshes already describing
                                 the physical wall need no such offset.
- `isovalue=0.5`:                Base contour level of the volume fraction field.
- `volume_tolerance_percent=0.1`: Relative volume (3D) or area (2D) tolerance of correction.
- `volume_max_iterations=8`:     Maximum isovalue-correction iterations.
- `minimum_isovalue=0.1`:        Lower correction bracket.
- `maximum_isovalue=0.9`:        Upper correction bracket.
- `warm_start=true`:             Seed the correction with the previous effective isovalue.
- `keep_enclosed_fluid=false`:   By default, fluid particles strictly inside boundary
                                 meshes are excluded before reconstruction: the boundary
                                 constraint removes their volume from the surface, so
                                 they must not count toward the volume-correction target
                                 either. `true` keeps them.
- `sparse_component_fallback=false`: Add volume-equivalent spheres (3D) or area-equivalent
                                 polygonal circles (2D) for compact
                                 source components that the primary level set does not
                                 resolve.
- `record_field_moments=false`:  Record intermediate field integrals in the returned stats.
- `parallelization_backend=nothing`: CPU threading backend for the reconstruction
                                 loops (e.g. `SerialBackend()` disables threading). With
                                 `nothing`, the backend of the simulation is used when
                                 reconstructing from a `Semidiscretization` (callback and
                                 system-based [`reconstruct_surface`](@ref)); GPU and
                                 KernelAbstractions backends map to `PolyesterBackend()`
                                 because the reconstruction runs on CPU copies of the data.
                                 Direct calls with particle arrays use `default_backend`.

# Examples
```jldoctest; output = false
# Production tank convention: lower corner at the origin, open top
reconstruction = SurfaceReconstruction(particle_spacing=0.05; tank_size=(1.0, 0.6, 1.0))

# output
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ SurfaceReconstruction                                                                            │
│ ═════════════════════                                                                            │
│ particle spacing: ………………………………… 0.05                                                             │
│ voxel size: ………………………………………………… 0.025                                                            │
│ Gaussian sigma (voxels): ……………… 1.8                                                              │
│ domain min corner: ……………………………… (0.0, 0.0, 0.0)                                                  │
│ domain max corner: ……………………………… (1.0, 0.6, 1.0)                                                  │
│ open faces: ………………………………………………… +y                                                               │
│ boundary clearance: …………………………… 0.0                                                              │
│ isovalue: ……………………………………………………… 0.5                                                              │
│ volume tolerance (%): ……………………… 0.1                                                              │
│ warm start: ………………………………………………… yes                                                              │
│ sparse component fallback: ………… no                                                               │
│ parallelization backend: ……………… inherited                                                        │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```
"""
function SurfaceReconstruction(; particle_spacing, ndims=nothing,
                               voxel_size=nothing,
                               gaussian_sigma=nothing,
                               gaussian_sigma_voxels=nothing,
                               tank_size=nothing,
                               min_corner=nothing, max_corner=nothing,
                               open_faces=nothing,
                               boundary_clearance::Real=0.0,
                               isovalue::Real=0.5,
                               volume_tolerance_percent::Real=0.1,
                               volume_max_iterations::Integer=8,
                               minimum_isovalue::Real=0.1,
                               maximum_isovalue::Real=0.9,
                               warm_start::Bool=true,
                               record_field_moments::Bool=false,
                               sparse_component_fallback::Bool=false,
                               keep_enclosed_fluid::Bool=false,
                               parallelization_backend=nothing)
    dimension = something(ndims,
                          tank_size !== nothing ? length(tank_size) :
                          min_corner !== nothing ? length(min_corner) : 3)
    dimension isa Integer && dimension in (2, 3) ||
        throw(ArgumentError("surface reconstruction supports integer dimensions 2 and 3"))
    (isfinite(particle_spacing) && particle_spacing > 0) ||
        throw(ArgumentError("`particle_spacing` must be finite and positive"))

    voxel_size_ = something(voxel_size, particle_spacing / 2)
    (isfinite(voxel_size_) && voxel_size_ > 0) ||
        throw(ArgumentError("`voxel_size` must be finite and positive"))

    if gaussian_sigma !== nothing && gaussian_sigma_voxels !== nothing
        throw(ArgumentError("specify only one Gaussian sigma representation"))
    end
    sigma_voxels = gaussian_sigma_voxels !== nothing ? gaussian_sigma_voxels :
                   something(gaussian_sigma, 0.9 * particle_spacing) / voxel_size_
    (isfinite(sigma_voxels) && sigma_voxels > 0) ||
        throw(ArgumentError("the Gaussian sigma must be finite and positive"))

    if tank_size !== nothing
        (min_corner === nothing && max_corner === nothing &&
         open_faces === nothing) ||
            throw(ArgumentError("`tank_size` cannot be combined with `min_corner`, `max_corner` or `open_faces`"))
        length(tank_size) == dimension ||
            throw(ArgumentError("tank dimensions do not match `ndims`"))
        tank_size_ = SVector{dimension, Float64}(Float64.(tank_size)...)
        all(isfinite, tank_size_) && all(tank_size_ .> 0) ||
            throw(ArgumentError("`tank_size` must be finite and positive"))
        min_corner_ = zero(SVector{dimension, Float64})
        max_corner_ = tank_size_
        # Production tank convention: only the +y face is open.
        open_faces_ = ntuple(face -> face == 4, 2dimension)
    elseif min_corner === nothing && max_corner === nothing
        open_faces !== nothing &&
            throw(ArgumentError("`open_faces` requires `min_corner` and `max_corner`"))
        min_corner_ = max_corner_ = nothing
        open_faces_ = ntuple(_ -> false, 2dimension)
    else
        (min_corner !== nothing && max_corner !== nothing) ||
            throw(ArgumentError("`min_corner` and `max_corner` must be given together"))
        length(min_corner) == length(max_corner) == dimension ||
            throw(ArgumentError("domain dimensions do not match `ndims`"))
        min_corner_ = SVector{dimension, Float64}(Float64.(min_corner)...)
        max_corner_ = SVector{dimension, Float64}(Float64.(max_corner)...)
        all(isfinite, min_corner_) && all(isfinite, max_corner_) ||
            throw(ArgumentError("`min_corner`/`max_corner` must be finite"))
        all(min_corner_ .< max_corner_) ||
            throw(ArgumentError("`min_corner` must be strictly below `max_corner`"))
        if open_faces === nothing
            open_faces_ = ntuple(_ -> false, 2dimension)
        else
            length(open_faces) == 2dimension ||
                throw(ArgumentError("`open_faces` needs $(2dimension) face flags"))
            open_faces_ = NTuple{2dimension, Bool}(open_faces)
        end
    end

    options = SurfaceReconstructionOptions(isovalue=Float64(isovalue),
                                           volume_tolerance_percent=Float64(volume_tolerance_percent),
                                           volume_max_iterations=Int(volume_max_iterations),
                                           minimum_isovalue=Float64(minimum_isovalue),
                                           maximum_isovalue=Float64(maximum_isovalue),
                                           warm_start=warm_start,
                                           record_field_moments=record_field_moments,
                                           sparse_component_fallback=sparse_component_fallback,
                                           keep_enclosed_fluid=keep_enclosed_fluid)
    isfinite(options.volume_tolerance_percent) && options.volume_tolerance_percent > 0 ||
        throw(ArgumentError("`volume_tolerance_percent` must be finite and positive"))
    options.volume_max_iterations > 0 ||
        throw(ArgumentError("`volume_max_iterations` must be positive"))
    isfinite(options.maximum_isovalue) &&
    0 < options.minimum_isovalue < options.maximum_isovalue ||
        throw(ArgumentError("isovalue search bounds must be finite, positive and increasing"))
    options.minimum_isovalue < options.isovalue < options.maximum_isovalue ||
        throw(ArgumentError("`isovalue` must be inside the correction interval"))
    isfinite(boundary_clearance) && boundary_clearance >= 0 ||
        throw(ArgumentError("`boundary_clearance` must be finite and nonnegative"))

    padding = 4 * particle_spacing

    parallelization_backend === nothing ||
        parallelization_backend isa PointNeighbors.AbstractThreadingBackend ||
        throw(ArgumentError("`parallelization_backend` must be `nothing` or a CPU threading backend"))

    return SurfaceReconstruction{dimension, 2dimension}(options, Float64(particle_spacing),
                                                        Float64(voxel_size_),
                                                        Float64(sigma_voxels), min_corner_,
                                                        max_corner_,
                                                        open_faces_,
                                                        Float64(boundary_clearance),
                                                        Float64(padding),
                                                        parallelization_backend,
                                                        SurfaceReconstructionCache(Ref{SurfaceWorkspace}(nothing),
                                                                                   Ref(Float64(isovalue)),
                                                                                   Ref{LastMeshAnalysis}(nothing)))
end

function Base.show(io::IO, reconstruction::SurfaceReconstruction)
    @nospecialize reconstruction # reduce precompilation time

    print(io, "SurfaceReconstruction(particle_spacing=", reconstruction.particle_spacing,
          ", voxel_size=", reconstruction.voxel_size, ")")
end

function Base.show(io::IO, ::MIME"text/plain", reconstruction::SurfaceReconstruction)
    @nospecialize reconstruction # reduce precompilation time

    if get(io, :compact, false)
        show(io, reconstruction)
    else
        (; options, min_corner, max_corner, open_faces) = reconstruction
        face_names = ("-x", "+x", "-y", "+y", "-z", "+z")[1:length(open_faces)]
        open_list = join(face_names[collect(open_faces)], ", ")
        setup = Pair{String, Any}["particle spacing" => reconstruction.particle_spacing,
                                  "voxel size" => reconstruction.voxel_size,
                                  "Gaussian sigma (voxels)" => reconstruction.sigma_voxels]
        if min_corner === nothing
            push!(setup, "domain" => "adaptive (unconstrained)")
        else
            push!(setup, "domain min corner" => Tuple(min_corner))
            push!(setup, "domain max corner" => Tuple(max_corner))
            push!(setup, "open faces" => isempty(open_list) ? "none" : open_list)
        end
        append!(setup,
                [
                    "boundary clearance" => reconstruction.boundary_clearance,
                    "isovalue" => options.isovalue,
                    "volume tolerance (%)" => options.volume_tolerance_percent,
                    "warm start" => options.warm_start ? "yes" : "no",
                    "sparse component fallback" => options.sparse_component_fallback ?
                                                   "yes" : "no",
                    "parallelization backend" => something(reconstruction.parallelization_backend,
                                                           "inherited")
                ])
        summary_box(io, "SurfaceReconstruction", setup)
    end
end

# The reconstruction runs on CPU data: CPU threading backends are used as they are, while
# GPU and KernelAbstractions backends map to the default CPU backend, consistent with
# `transfer2cpu(semi)`.
cpu_threading_backend(backend::PointNeighbors.AbstractThreadingBackend) = backend
cpu_threading_backend(backend) = PolyesterBackend()

# Same configuration and shared workspace buffers, but an independent warm-start state.
# Used when one callback reconstructs several fluid systems, so that one system's
# effective isovalue does not seed another system's correction.
function independent_warm_start(reconstruction::SurfaceReconstruction)
    cache = SurfaceReconstructionCache(reconstruction.cache.workspace,
                                       Ref(reconstruction.options.isovalue),
                                       Ref{LastMeshAnalysis}(nothing))
    return @set reconstruction.cache = cache
end

"""
    reconstruct_surface!(reconstruction, points, volumes, boundaries=[];
                         initial_isovalue=nothing, parallelization_backend=nothing)

Reconstruct a closed free surface from `points` (2×n or 3×n matrix), `volumes`
(per-particle areas in 2D or volumes in 3D), and optional `boundaries` built with
`BoundaryMesh`, reusing the workspace and warm-start state
of `reconstruction`. Returns the reconstructed `SurfaceMesh` and a
[`SurfaceReconstructionStatistics`](@ref).

The threading backend is the one configured in `reconstruction`; if that is `nothing`,
the `parallelization_backend` keyword (e.g. the simulation's backend), and otherwise
`default_backend(points)`.
"""
function reconstruct_surface!(reconstruction::SurfaceReconstruction, points, volumes,
                              boundaries=(); initial_isovalue=nothing,
                              parallelization_backend=nothing)
    (; options, particle_spacing, voxel_size, sigma_voxels, min_corner, max_corner,
     open_faces, boundary_clearance, grid_padding, cache) = reconstruction

    backend = cpu_threading_backend(something(reconstruction.parallelization_backend,
                                              parallelization_backend,
                                              default_backend(points)))

    ndims(points) == 2 && size(points, 1) == ndims(reconstruction) ||
        throw(ArgumentError("`points` must be a $(ndims(reconstruction))×n matrix of particle coordinates"))
    length(volumes) == size(points, 2) ||
        throw(DimensionMismatch("`volumes` length does not match the particle count"))
    isempty(volumes) && throw(ArgumentError("surface reconstruction needs particles"))
    start_isovalue = initial_isovalue !== nothing ? Float64(initial_isovalue) :
                     options.warm_start ? cache.previous_isovalue[] : options.isovalue
    isfinite(start_isovalue) &&
    options.minimum_isovalue <= start_isovalue <= options.maximum_isovalue ||
        throw(ArgumentError("`initial_isovalue` must be finite and within the correction interval"))
    # Explicit loop (instead of `all`) so that large static matrices do not
    # trigger compile-time unrolling.
    for value in points
        isfinite(value) ||
            throw(ArgumentError("`points` contains non-finite coordinates"))
    end
    for volume in volumes
        (isfinite(volume) && volume > 0) ||
            throw(ArgumentError("`volumes` must be finite and positive"))
    end

    # Fluid inside boundaries is clipped away by the boundary constraint; exclude it from
    # the volume-correction target as well (production behavior), unless requested not to.
    boundaries = collect(boundaries)
    all(boundary -> boundary isa BoundaryMesh && ndims(boundary) == ndims(reconstruction),
        boundaries) ||
        throw(ArgumentError("boundary dimensions must match the reconstruction"))
    enclosed_count, enclosed_volume = 0, 0.0
    if !options.keep_enclosed_fluid && !isempty(boundaries)
        enclosed = enclosed_particles(points, boundaries; backend)
        enclosed_count = count(!iszero, enclosed)
        if enclosed_count > 0
            enclosed_volume = sum(volumes[index]
                                  for index in eachindex(volumes)
                                  if !iszero(enclosed[index]))
            retained = findall(iszero, enclosed)
            isempty(retained) &&
                throw(ArgumentError("all particles lie inside boundaries"))
            points = points[:, retained]
            volumes = volumes[retained]
        end
    end

    grid,
    domain = reconstruction_grid(points, voxel_size, grid_padding;
                                 min_corner=min_corner, max_corner=max_corner,
                                 open_faces=open_faces)

    points, volumes, excluded_count,
    excluded_volume, support = exclude_and_support(points, volumes, grid)
    if excluded_count > 0
        @warn "$excluded_count particles (volume $excluded_volume) lie outside the " *
              "reconstruction grid and are excluded; enlarge the domain or open the " *
              "corresponding faces" maxlog=10
        isempty(volumes) &&
            throw(ArgumentError("all particles lie outside the reconstruction grid"))
    end

    # Reuse the workspace only for the identical grid: the marching-cubes coordinates
    # are baked into the workspace at construction.
    if cache.workspace[] === nothing ||
       size(cache.workspace[].field) != grid.dimensions ||
       cache.workspace[].origin != grid.origin || cache.workspace[].spacing != grid.spacing
        cache.workspace[] = ReconstructionWorkspace(grid; backend)
    end
    workspace = cache.workspace[]
    # The backend can change between calls (e.g. a reused reconstruction); the grid
    # buffers do not depend on it.
    workspace.backend = backend

    mesh,
    stats,
    mesh_analysis = @trixi_timeit timer() "reconstruct surface" begin
        _reconstruct!(workspace, points, volumes,
                      boundaries, grid.origin, voxel_size,
                      domain, particle_spacing,
                      sigma_voxels, boundary_clearance, support,
                      options; initial_isovalue=start_isovalue)
    end
    cache.previous_isovalue[] = stats["effective_isovalue"]
    # Hand the liquid analysis to the surface writers via the cache. It describes the
    # returned mesh object exactly when present.
    cache.last_mesh_analysis[] = isnothing(mesh_analysis) ? nothing : (mesh, mesh_analysis)

    details = statistics_dict(stats)
    details["n_excluded_particles"] = excluded_count
    details["excluded_particle_volume"] = excluded_volume
    details["n_enclosed_particles"] = enclosed_count
    details["enclosed_particle_volume"] = enclosed_volume

    return mesh, stats
end

"""
    reconstruct_surface(points, volumes, boundaries=[]; particle_spacing, kwargs...)

One-shot allocation of a fresh [`SurfaceReconstruction`](@ref) followed by
[`reconstruct_surface!`](@ref). All `kwargs` are forwarded to the
`SurfaceReconstruction` constructor. For repeated reconstruction of simulation frames,
create the `SurfaceReconstruction` once and call `reconstruct_surface!` to reuse its
workspace.
"""
function reconstruct_surface(points, volumes, boundaries=(); ndims=size(points, 1),
                             kwargs...)
    reconstruction = SurfaceReconstruction(; ndims=something(ndims, size(points, 1)),
                                           kwargs...)

    return reconstruct_surface!(reconstruction, points, volumes, boundaries)
end

function system_index_spec(spec::Integer, semi)
    1 <= spec <= length(semi.systems) ||
        throw(ArgumentError("system index $spec is out of range"))
    return spec
end

function system_index_spec(spec::AbstractSystem, semi)
    return system_indices(spec, semi)
end

function system_index_spec(spec, semi)
    throw(ArgumentError("expected a system index or a system, got a `$(typeof(spec))`"))
end

"""
    particle_volumes(system, v)

Per-particle fluid volumes `Vᵢ = mᵢ / ρᵢ` for all active particles of a fluid system:
the hydrodynamic (constant) particle masses divided by the current densities. Inactive
buffer particles (e.g. of `OpenBoundarySystem`) are excluded.
"""
function particle_volumes(system::AbstractFluidSystem, v)
    density = current_density(v, system)
    return [Float64(hydrodynamic_mass(system, particle)) / Float64(density[particle])
            for particle in each_active_particle(system)]
end

"""
    active_surface_points(system, u)

Current coordinates of all active particles as a matrix with one row per dimension,
avoiding a copy when the coordinates already have that layout. Inactive buffer particles are
excluded.
"""
function active_surface_points(system, u)
    coordinates = active_coordinates(u, system)
    return coordinates isa Matrix{Float64} ? coordinates :
           Matrix{Float64}(coordinates)
end

"""
    reconstruct_surface(system, v_ode, u_ode, semi; particle_spacing=nothing, kwargs...)

One-shot reconstruction of the free surface of a fluid `system` from live ODE vectors,
e.g. `sol.u[end].x`. Only active particles contribute. The particle spacing defaults to
the system's spacing; all other `kwargs` are forwarded to [`SurfaceReconstruction`](@ref).

For `SummationDensity`, the density cache belongs to the last right-hand-side evaluation.
Call `update_systems_and_nhs(v_ode, u_ode, semi, t)` first when the vectors do not come
from the final state of a solve (the [`reconstruct_surface(semi, sol)`](@ref) method does
this automatically).
"""
function reconstruct_surface(system::AbstractFluidSystem, v_ode, u_ode, semi;
                             particle_spacing=nothing, ndims=Base.ndims(system), kwargs...)
    ndims == Base.ndims(system) && ndims in (2, 3) ||
        throw(ArgumentError("reconstruction dimensions must match the 2D or 3D fluid system"))

    # Work on CPU copies (a no-op on the CPU) so offline reconstruction also works
    # directly with GPU state.
    v_ode, u_ode, system, semi = transfer2cpu(v_ode, u_ode, system, semi)

    isempty(each_active_particle(system)) &&
        throw(ArgumentError("system has no active particles to reconstruct"))
    # Field access instead of the `particle_spacing` accessor: the keyword of the same
    # name would shadow the function inside this body.
    spacing = something(particle_spacing,
                        Float64(system.initial_condition.particle_spacing))
    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)
    points = active_surface_points(system, u)
    volumes = particle_volumes(system, v)
    reconstruction = SurfaceReconstruction(; particle_spacing=spacing, ndims, kwargs...)

    return reconstruct_surface!(reconstruction, points, volumes;
                                parallelization_backend=semi.parallelization_backend)
end

"""
    reconstruct_surface(semi, sol; system=nothing, frame=lastindex(sol.u), kwargs...)

One-shot reconstruction from a solved `semi` at `sol.u[frame]`. `system` selects the
fluid system (index or object; default: the first fluid system). All `kwargs` are
forwarded to [`reconstruct_surface`](@ref) for systems.

The particle data is taken from the semidiscretization the solution was computed with
(`sol.prob.p.semi`). On the CPU, this is `semi` itself; for GPU simulations,
`semidiscretize` works on a device copy, while `semi` only identifies the systems.
"""
function reconstruct_surface(semi, sol::ODESolution; system=nothing,
                             frame=lastindex(sol.u), kwargs...)
    systems = semi.systems
    if system === nothing
        system_index = findfirst(system -> system isa AbstractFluidSystem, systems)
        system_index === nothing &&
            throw(ArgumentError("no fluid system found; specify `system`"))
    else
        system_index = system_index_spec(system, semi)
        systems[system_index] isa AbstractFluidSystem ||
            throw(ArgumentError("system $system_index is not a fluid system"))
    end
    v_ode, u_ode = sol.u[frame].x

    # The solution vectors belong to the semidiscretization of the ODE problem, which is a
    # device copy of `semi` for GPU simulations
    solution_semi = sol.prob.p.semi
    map(nparticles, solution_semi.systems) == map(nparticles, systems) ||
        throw(ArgumentError("`sol` was not computed with `semi`"))

    # Refresh system caches (e.g. the `SummationDensity` cache) to the frame's state;
    # they otherwise hold the values of the last right-hand-side evaluation.
    update_systems_and_nhs(v_ode, u_ode, solution_semi, sol.t[frame])

    return reconstruct_surface(solution_semi.systems[system_index], v_ode, u_ode,
                               solution_semi; kwargs...)
end

"""
    reconstruct_surface(initial_condition::InitialCondition; particle_spacing, kwargs...)

One-shot reconstruction from an [`InitialCondition`](@ref), e.g. produced by
[`vtk2trixi`](@ref), with volumes from the stored masses and densities. The particle
spacing defaults to the initial condition's spacing (which must be positive).
"""
function reconstruct_surface(initial_condition::InitialCondition;
                             particle_spacing=initial_condition.particle_spacing,
                             ndims=size(initial_condition.coordinates, 1),
                             kwargs...)
    (isfinite(particle_spacing) && particle_spacing > 0) ||
        throw(ArgumentError("`particle_spacing` must be finite and positive; " *
                            "the initial condition does not define one"))
    points = active_surface_points(initial_condition, initial_condition.coordinates)
    volumes = Vector{Float64}(initial_condition.mass) ./
              Vector{Float64}(initial_condition.density)
    reconstruction = SurfaceReconstruction(; particle_spacing=Float64(particle_spacing),
                                           ndims, kwargs...)

    return reconstruct_surface!(reconstruction, points, volumes)
end

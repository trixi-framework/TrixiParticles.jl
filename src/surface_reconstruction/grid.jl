# Reconstruction grid, workspace buffers, and field utilities.
"""
    ReconstructionGrid(origin, spacing, dimensions)

Uniform grid for 2D or 3D reconstruction. `origin` is the first grid node and `spacing`
the distance between nodes. See [`reconstruction_grid`](@ref) for
construction from particle bounds or a fixed tank.
"""
struct ReconstructionGrid{NDIMS}
    origin::SVector{NDIMS, Float64}
    spacing::Float64
    dimensions::Dims{NDIMS}
end

function ReconstructionGrid(origin::AbstractVector, spacing::Real,
                            dimensions::NTuple{N, Int}) where {N}
    return ReconstructionGrid{N}(SVector{N, Float64}(origin), Float64(spacing), dimensions)
end

"""
    reconstruction_grid(points, voxel_size, nominal_padding; min_corner=nothing, max_corner=nothing,
                        open_faces=(false, false, false, false, false, false))

Derive the reconstruction grid and the clipping domain from particle bounds. With
`min_corner`/`max_corner` (e.g. resolved from `tank_size`), the grid is pinned to
`[min - padding, max + padding]` and the domain is the given box with `open_faces`
(`(-x, +x, -y, +y[, -z, +z])` order). Without corners, the grid adapts to the particle
bounds with the same padding, and the domain is the padded grid box itself, so the
reconstruction is translation-invariant and effectively unconstrained. Returns the grid
and the [`ReconstructionDomain`](@ref).
"""
function reconstruction_grid(points, voxel_size, nominal_padding; min_corner=nothing,
                             max_corner=nothing,
                             open_faces=ntuple(_ -> false, 2size(points, 1)))
    # Keep the two supported dimensions visible to inference at this entry point;
    # the common implementation specializes its tuples and static vectors on N.
    if size(points, 1) == 2
        return reconstruction_grid(points, voxel_size, nominal_padding, Val(2);
                                   min_corner, max_corner, open_faces)
    end
    return reconstruction_grid(points, voxel_size, nominal_padding, Val(3);
                               min_corner, max_corner, open_faces)
end

function reconstruction_grid(points, voxel_size, nominal_padding, ::Val{N};
                             min_corner, max_corner, open_faces) where {N}
    padding = nominal_padding + voxel_size / 2
    if min_corner !== nothing
        lower = SVector{N, Float64}(min_corner)
        upper_corner = SVector{N, Float64}(max_corner)
        origin = lower .- padding
        upper = upper_corner .+ padding
        domain = ReconstructionDomain(lower, upper_corner; open_faces=open_faces)
        if any(open_faces) && size(points, 2) > 0
            origin,
            upper = extend_over_open_faces(points, origin, upper, padding,
                                           voxel_size, open_faces)
        end
    else
        minimums = SVector{N, Float64}(ntuple(axis -> minimum(view(points, axis, :)),
                                              Val(N)))
        maximums = SVector{N, Float64}(ntuple(axis -> maximum(view(points, axis, :)),
                                              Val(N)))
        origin = minimums .- padding
        upper = maximums .+ padding
        # Without explicit corners, only clip the surface at the padded grid box itself
        # (closed on all faces), which keeps the reconstruction translation-invariant.
        domain = ReconstructionDomain(origin, upper)
    end
    dimensions = Tuple(ceil.(Int, (upper - origin) ./ voxel_size) .+ 1)

    return ReconstructionGrid(origin, voxel_size, dimensions), domain
end

# Open faces are not walls: water may leave the nominal box through them (e.g. splashes
# above an open tank) and must still be reconstructed. Extend the grid over open faces to
# cover the particles while keeping the voxel phase of the nominal grid (lower extensions
# shift the origin by whole voxels). Extensions are capped at one nominal extent per side
# to bound memory for runaway particles; the caller excludes particles beyond the cap.
function extend_over_open_faces(points, origin, upper, padding, voxel_size, open_faces)
    extent = upper - origin
    for axis in eachindex(origin)
        coordinates = view(points, axis, :)
        if open_faces[2axis - 1]
            needed = origin[axis] - (minimum(coordinates) - padding)
            if needed > 0
                steps = ceil(min(needed, extent[axis]) / voxel_size)
                origin = setindex(origin, origin[axis] - steps * voxel_size, axis)
            end
        end
        if open_faces[2axis]
            needed = maximum(coordinates) + padding - upper[axis]
            if needed > 0
                upper = setindex(upper, upper[axis] + min(needed, extent[axis]), axis)
            end
        end
    end
    return origin, upper
end

# Planar reconstruction needs no marching-cubes buffers. Full 2D field passes are small
# and keep workspace reuse independent of the previous frame's occupied region.
mutable struct ReconstructionWorkspace2D
    field::Matrix{Float32}
    temporary::Matrix{Float32}
    scratch::Matrix{Float32}
    constraint::Matrix{Float32}
    backend::PointNeighbors.AbstractThreadingBackend
    origin::SVector{2, Float64}
    spacing::Float64
end

# Particles whose bilinear/trilinear CIC stencil does not fit cannot be deposited.
# Excluding them from both the deposit and the volume-correction target keeps the two
# consistent; silently dropping stencil weights would let the correction inflate the
# remaining surface to absorb volume that is not on the grid. The index arithmetic
# mirrors `deposit_volume_cic!` exactly.
function exclude_outside_grid(points, volumes, grid::ReconstructionGrid{N}) where {N}
    (; origin, spacing, dimensions) = grid
    inverse_spacing = inv(spacing)
    inside = trues(size(points, 2))
    for particle in axes(points, 2), axis in 1:N
        lower = floor(Int, (points[axis, particle] - origin[axis]) * inverse_spacing)
        if !(0 <= lower <= dimensions[axis] - 2)
            inside[particle] = false
        end
    end

    excluded_count = count(!, inside)
    excluded_count == 0 && return points, volumes, 0, 0.0

    excluded_volume = sum(volumes[.!inside])
    return points[:, inside], volumes[inside], excluded_count, excluded_volume
end

# Single-pass fusion of `exclude_outside_grid` and `deposition_support` for the
# reconstruction pipeline: one loop over particles computes the inside mask and the
# min/max stencil cells of the retained particles. The per-particle index arithmetic is
# unchanged, and `min`/`max` over integers are exact and order-independent, so the mask,
# the filtered data, and the support ranges are identical to the two separate passes.
@inline stencil_fits(::Tuple{}, ::Tuple{}) = true
@inline function stencil_fits(cell::Tuple, dimensions::Tuple)
    return 0 <= first(cell) <= first(dimensions) - 2 &&
           stencil_fits(Base.tail(cell), Base.tail(dimensions))
end

function exclude_and_support(points, volumes, grid::ReconstructionGrid{N}) where {N}
    inside = trues(size(points, 2))
    lower, upper = particle_stencil_bounds!(inside, points, grid)
    # 0-based stencil cells `cell` and `cell + 1` are the 1-based voxels `cell + 1:cell + 2`.
    # Map the final tuples as arguments; capturing reassigned loop accumulators in a
    # closure would box them and cause per-particle allocations.
    support = lower[1] == typemax(Int) ? ntuple(_ -> 1:0, Val(N)) :
              map((low, high) -> (low + 1):(high + 2), lower, upper)

    excluded_count = count(!, inside)
    excluded_count == 0 && return points, volumes, 0, 0.0, support

    excluded_volume = sum(volumes[.!inside])
    return points[:, inside], volumes[inside], excluded_count, excluded_volume, support
end

# `inside` starts filled with true. Only the numerical scan is specialized below;
# mask filtering, excluded measure, and support construction are shared.
@inline function particle_stencil_bounds!(inside, points,
                                          grid::ReconstructionGrid{N}) where {N}
    (; origin, spacing, dimensions) = grid
    inverse_spacing = inv(spacing)
    lower = ntuple(_ -> typemax(Int), Val(N))
    upper = ntuple(_ -> typemin(Int), Val(N))
    @inbounds for particle in axes(points, 2)
        # The caller checks the coordinate dimension. Mark indexing inside the closure
        # too: the outer @inbounds does not propagate through both ntuple and its lambda.
        cell = ntuple(axis -> @inbounds(floor(Int,
                                              (points[axis, particle] - origin[axis]) *
                                              inverse_spacing)), Val(N))
        if stencil_fits(cell, dimensions)
            lower = map(min, lower, cell)
            upper = map(max, upper, cell)
        else
            inside[particle] = false
        end
    end
    return lower, upper
end

# The scalar 3D scan benchmarks faster than the tuple scan at production scale. Keep
# its register/branch layout while sharing the surrounding logic with planar inputs.
@inline function particle_stencil_bounds!(inside, points, grid::ReconstructionGrid{3})
    (; origin, spacing, dimensions) = grid
    inverse_spacing = inv(spacing)
    nx, ny, nz = dimensions
    lower_x, lower_y, lower_z = typemax(Int), typemax(Int), typemax(Int)
    upper_x, upper_y, upper_z = typemin(Int), typemin(Int), typemin(Int)
    @inbounds for particle in axes(points, 2)
        cell_x = floor(Int, (points[1, particle] - origin[1]) * inverse_spacing)
        cell_y = floor(Int, (points[2, particle] - origin[2]) * inverse_spacing)
        cell_z = floor(Int, (points[3, particle] - origin[3]) * inverse_spacing)
        if 0 <= cell_x <= nx - 2 && 0 <= cell_y <= ny - 2 && 0 <= cell_z <= nz - 2
            lower_x = min(lower_x, cell_x)
            lower_y = min(lower_y, cell_y)
            lower_z = min(lower_z, cell_z)
            upper_x = max(upper_x, cell_x)
            upper_y = max(upper_y, cell_y)
            upper_z = max(upper_z, cell_z)
        else
            inside[particle] = false
        end
    end
    return (lower_x, lower_y, lower_z), (upper_x, upper_y, upper_z)
end

mutable struct ReconstructionWorkspace
    field::Array{Float32, 3}
    temporary::Array{Float32, 3}
    scratch::Array{Float32, 3}
    constraint::Array{Float32, 3}
    backend::PointNeighbors.AbstractThreadingBackend
    origin::SVector{3, Float64}
    spacing::Float64
    # Denormalization of marching-cubes vertices from grid-index space, computed exactly
    # as MarchingCubes.jl does for the full grid (see `marching_cubes_denormalization`)
    vertex_offset::SVector{3, Float32}
    vertex_scale::SVector{3, Float32}
    # Reusable buffers of the box-restricted marching cubes; they only grow
    vertex_indices::Vector{Int32}
    cell_cases::Vector{UInt8}
    # `field`, `temporary`, and `scratch` are zero outside this region, which bounds all
    # writes of the last reconstruction (see `_reconstruct!`)
    dirty_region::NTuple{3, UnitRange{Int}}
    # `constraint` holds the domain constraint for this key (domain, origin, voxel size,
    # distance scale), modified by boundary constraints only in `boundary_regions`
    constraint_key::Union{Nothing,
                          Tuple{ReconstructionDomain, SVector{3, Float64}, Float64,
                                Float64}}
    boundary_regions::Vector{NTuple{3, UnitRange{Int}}}
end

function ReconstructionWorkspace(dimensions, origin, spacing;
                                 backend=PolyesterBackend())
    field = zeros(Float32, dimensions)
    temporary = similar(field)
    scratch = similar(field)
    constraint = similar(field)
    vertex_offset,
    vertex_scale = marching_cubes_denormalization(dimensions, origin,
                                                  spacing)
    # `temporary` and `scratch` are uninitialized
    dirty_region = full_region(size(field))
    return ReconstructionWorkspace(field, temporary, scratch, constraint, backend,
                                   SVector{3, Float64}(origin), Float64(spacing),
                                   vertex_offset,
                                   vertex_scale, Int32[], UInt8[], dirty_region, nothing,
                                   NTuple{3, UnitRange{Int}}[])
end

function full_region(dimensions::NTuple{N, Int}) where {N}
    ntuple(axis -> 1:dimensions[axis], Val(N))
end

# `region` grown by `layers` voxels in every direction, clamped to the grid
function expand_region(region, layers, dimensions::NTuple{N, Int}) where {N}
    return ntuple(axis -> max(first(region[axis]) - layers,
                              1):min(last(region[axis]) + layers,
                                     dimensions[axis]), Val(N))
end

# Deterministic parallel reductions over a 3D field. Partial results per slab of constant
# `k` (Float64 accumulation in memory order) are combined in a fixed order, so the results
# do not depend on the number of threads. Minimum and maximum are exact.
# If the field is known to be zero outside `region`, only `region` is traversed. The
# partial sums then skip exact zeros only, so the results are bitwise identical to the
# full traversal.
function field_sum_extrema(field; backend=PolyesterBackend(), region=axes(field))
    xs, ys, zs = region
    nz = size(field, 3)
    sums = zeros(Float64, nz)
    minima = fill(typemax(eltype(field)), nz)
    maxima = fill(typemin(eltype(field)), nz)
    @threaded backend for k in zs
        slab_sum = 0.0
        slab_min, slab_max = typemax(eltype(field)), typemin(eltype(field))
        @inbounds for j in ys, i in xs
            value = field[i, j, k]
            slab_sum += value
            slab_min = min(slab_min, value)
            slab_max = max(slab_max, value)
        end
        @inbounds sums[k], minima[k], maxima[k] = slab_sum, slab_min, slab_max
    end
    field_min, field_max = minimum(minima), maximum(maxima)
    if region != axes(field)
        # The zeros outside `region`
        field_min = min(field_min, zero(field_min))
        field_max = max(field_max, zero(field_max))
    end
    return sum(sums), field_min, field_max
end

function parallel_fill!(field, value; backend=PolyesterBackend(), region=axes(field))
    xs, ys, zs = region
    @threaded backend for k in zs
        @inbounds for j in ys, i in xs
            field[i, j, k] = value
        end
    end
    return field
end

# `MarchingCubes.denormalize` maps grid-index-space vertices to coordinates with the
# extrema of the (Float32) grid coordinates. Reproduce its arithmetic exactly, so that the
# box-restricted port yields bitwise the same vertices as the library on the full grid.
function marching_cubes_denormalization(dimensions, origin, spacing)
    x, y,
    z = ntuple(axis -> Float32.(range(origin[axis]; step=spacing,
                                      length=dimensions[axis])), 3)
    mx, Mx = extrema(x)
    my, My = extrema(y)
    mz, Mz = extrema(z)
    nx, ny, nz = dimensions
    scale = @SVector([Mx - mx, My - my, Mz - mz]) ./ @SVector([nx - 1, ny - 1, nz - 1])
    offset = @SVector([mx, my, mz])
    return offset, scale
end

function ReconstructionWorkspace(grid::ReconstructionGrid;
                                 backend=PolyesterBackend())
    return ReconstructionWorkspace(grid.dimensions, grid.origin, grid.spacing;
                                   backend=backend)
end

function ReconstructionWorkspace(grid::ReconstructionGrid{2}; backend=PolyesterBackend())
    field = zeros(Float32, grid.dimensions)
    return ReconstructionWorkspace2D(field, similar(field), similar(field), similar(field),
                                     backend, grid.origin, grid.spacing)
end

@inline function trilinear_field_value(field, point, origin, spacing)
    coordinates = (point - origin) / spacing
    lower = floor.(Int, coordinates)
    fractions = coordinates - lower
    nx, ny, nz = size(field)
    0 <= lower[1] < nx - 1 && 0 <= lower[2] < ny - 1 &&
    0 <= lower[3] < nz - 1 ||
        error("particle center lies outside the reconstruction grid")
    value = 0.0
    @inbounds for z_offset in 0:1, y_offset in 0:1, x_offset in 0:1
        weight = (x_offset == 0 ? 1 - fractions[1] : fractions[1]) *
                 (y_offset == 0 ? 1 - fractions[2] : fractions[2]) *
                 (z_offset == 0 ? 1 - fractions[3] : fractions[3])
        value += weight * field[lower[1] + x_offset + 1,
                       lower[2] + y_offset + 1,
                       lower[3] + z_offset + 1]
    end
    return value
end

function scalar_field_moments(field, origin, spacing)
    total = 0.0
    first_x = 0.0
    first_y = 0.0
    first_z = 0.0
    @inbounds for z_index in axes(field, 3)
        z = origin[3] + (z_index - 1) * spacing
        for y_index in axes(field, 2)
            y = origin[2] + (y_index - 1) * spacing
            for x_index in axes(field, 1)
                value = Float64(field[x_index, y_index, z_index])
                x = origin[1] + (x_index - 1) * spacing
                total += value
                first_x += value * x
                first_y += value * y
                first_z += value * z
            end
        end
    end
    cell_volume = spacing^3
    integral = total * cell_volume
    first_moment = SVector{3, Float64}(first_x, first_y, first_z) * cell_volume
    centroid = integral > 0 ? first_moment / integral : SVector{3, Float64}(NaN, NaN, NaN)
    return (integral=integral,
            first_moment=first_moment,
            centroid=centroid)
end

function record_field_moments!(history, stage, field, origin, spacing)
    moments = scalar_field_moments(field, origin, spacing)
    push!(history,
          Dict(
              "stage" => stage,
              "integral" => moments.integral,
              "first_moment" => collect(moments.first_moment),
              "centroid" => collect(moments.centroid)
          ))
    return history
end

# Preserve the per-dimension summation/normalization order for diagnostic moments.
function scalar_field_moments(field::AbstractMatrix, origin, spacing)
    total, moment = 0.0, SVector(0.0, 0.0)
    for j in axes(field, 2), i in axes(field, 1)
        value = Float64(field[i, j])
        total += value
        moment += value * (origin + spacing * SVector(i - 1, j - 1))
    end
    return (; integral=total * spacing^2, first_moment=moment * spacing^2,
            centroid=total > 0 ? moment / total : SVector(NaN, NaN))
end

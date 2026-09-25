# Marching-cubes contouring and safeguarded isovalue volume correction.
function march_surface!(mc, isovalue; topology_aware=false)
    # MarchingCubes 0.1.11 does not clear this cache. Stale positive indices can
    # displace interior vertices in the topology-aware case table.
    fill!(mc.vert_indices, zero(eltype(mc.vert_indices)))
    if topology_aware
        # Experimental only. MarchingCubes 0.1.11 uses absolute eps(Float32) tests on
        # products of scalar values in its face/interior deciders. Low-amplitude fields
        # can consequently produce nonmanifold edges even with this stale-index fix.
        # See the analytic-saddle regression in `test/surface_reconstruction/correctness.jl`.
        MarchingCubes.march(mc, isovalue)
    else
        MarchingCubes.march_legacy(mc, isovalue)
    end
    return mc
end

# `region` must contain all voxels that marching cubes can classify as positive, expanded
# by two layers: marching cubes reads the active box (the positive voxels expanded by one
# layer) and the next layer above it, so it only reads scalars computed in `region`.
function contour!(workspace, isovalue; allow_empty=false,
                  region=axes(workspace.scratch))
    threshold = Float32(isovalue)
    box, any_positive = scalar_and_active_box!(workspace, threshold; region)
    mesh = box === nothing ? nothing : marching_cubes_box!(workspace, box)
    if mesh === nothing || isempty(mesh.vertices)
        allow_empty && !any_positive && return nothing
        error("surface reconstruction produced an empty mesh")
    end
    return mesh
end

# Marching cubes classifies a grid value `c` (relative to the isovalue 0) as positive iff
# `c > 0` after replacing `|c| < eps` by `eps`, i.e. iff `c > -eps`.
@inline marching_cubes_clamp(c::Float32) = abs(c) < eps(Float32) ? eps(Float32) : c
@inline marching_cubes_positive(c::Float32) = c > -eps(Float32)

# Compute the contouring scalar `min(field - threshold, constraint)` in `region` and,
# fused into the same pass, the bounding box of all values that marching cubes classifies
# as positive. Cells outside this box expanded by one layer have no sign change and
# produce neither vertices nor triangles. Returns the box (or `nothing`) and whether any
# value is `> 0`. See `contour!` for the requirements on `region`.
function scalar_and_active_box!(workspace, threshold; region=axes(workspace.scratch))
    (; field, constraint, backend) = workspace
    scalar = workspace.scratch
    nx, ny, nz = size(scalar)
    region_i, region_j, region_k = region
    lower_i = fill(typemax(Int), nz)
    upper_i = zeros(Int, nz)
    lower_j = fill(typemax(Int), nz)
    upper_j = zeros(Int, nz)
    # BitVector writes to different bits of the same word would race between slabs.
    positive = fill(false, nz)
    @threaded backend for k in region_k
        slab_lower_i, slab_upper_i = typemax(Int), 0
        slab_lower_j, slab_upper_j = typemax(Int), 0
        slab_positive = false
        @inbounds for j in region_j, i in region_i
            value = min(field[i, j, k] - threshold, constraint[i, j, k])
            scalar[i, j, k] = value
            if marching_cubes_positive(value)
                slab_lower_i = min(slab_lower_i, i)
                slab_upper_i = max(slab_upper_i, i)
                slab_lower_j = min(slab_lower_j, j)
                slab_upper_j = max(slab_upper_j, j)
            end
            slab_positive |= value > 0
        end
        @inbounds begin
            lower_i[k], upper_i[k] = slab_lower_i, slab_upper_i
            lower_j[k], upper_j[k] = slab_lower_j, slab_upper_j
            positive[k] = slab_positive
        end
    end

    any_positive = any(positive)
    occupied = findall(k -> upper_i[k] > 0, 1:nz)
    isempty(occupied) && return nothing, any_positive

    # Expand by one layer: sign-change edges and cells have a positive endpoint/corner
    i0 = max(minimum(lower_i[occupied]) - 1, 1)
    i1 = min(maximum(upper_i[occupied]) + 1, nx)
    j0 = max(minimum(lower_j[occupied]) - 1, 1)
    j1 = min(maximum(upper_j[occupied]) + 1, ny)
    k0 = max(first(occupied) - 1, 1)
    k1 = min(last(occupied) + 1, nz)
    return (i0:i1, j0:j1, k0:k1), any_positive
end

# Classic marching cubes (the `march_legacy` algorithm of MarchingCubes.jl without vertex
# normals), restricted to the active box and parallel over slabs of constant `k`.
# The result is bitwise identical to `MarchingCubes.march_legacy` on the full grid:
# vertices and triangles are emitted in the same (k, j, i) order, with slab-local counts
# and prefix sums placing every slab's output at its serial position, and vertex
# positions use the same full-grid index arithmetic and denormalization.
function marching_cubes_box!(workspace, box)
    (; backend, vertex_offset, vertex_scale) = workspace
    scalar = workspace.scratch
    nx, ny, nz = size(scalar)
    irange, jrange, krange = box
    i0, j0, k0 = first(irange), first(jrange), first(krange)
    bx, by, bz = length(irange), length(jrange), length(krange)

    # Box-local vertex ids per voxel and edge axis (x, y, z), zeroed per evaluation
    n_vertex_slots = 3 * bx * by * bz
    length(workspace.vertex_indices) < n_vertex_slots &&
        resize!(workspace.vertex_indices, n_vertex_slots)
    vertex_indices = workspace.vertex_indices
    @inline slot(axis, i, j, k) = axis + 3 * ((i - i0) + bx * ((j - j0) + by * (k - k0)))

    # Pass 1: count the vertices of every voxel slab
    vertex_counts = zeros(Int, bz)
    @threaded backend for k in krange
        count = 0
        @inbounds for j in jrange, i in irange
            c0 = marching_cubes_clamp(scalar[i, j, k])
            c1 = marching_cubes_clamp(i < nx ? scalar[i + 1, j, k] : scalar[i, j, k])
            c2 = marching_cubes_clamp(j < ny ? scalar[i, j + 1, k] : scalar[i, j, k])
            c3 = marching_cubes_clamp(k < nz ? scalar[i, j, k + 1] : scalar[i, j, k])
            if c0 < 0
                count += (c1 > 0) + (c2 > 0) + (c3 > 0)
            else
                count += (c1 < 0) + (c2 < 0) + (c3 < 0)
            end
        end
        vertex_counts[k - k0 + 1] = count
    end
    vertex_starts = cumsum(vertex_counts) .- vertex_counts
    vertices = Vector{SVector{3, Float32}}(undef, sum(vertex_counts))

    # Pass 2: emit vertices at their serial positions (full-grid index arithmetic as in
    # `add_x_vertex`, `add_y_vertex`, `add_z_vertex`) and record their ids per edge
    @threaded backend for k in krange
        next = vertex_starts[k - k0 + 1]
        @inbounds for j in jrange, i in irange
            base = slot(1, i, j, k)
            vertex_indices[base] = 0
            vertex_indices[base + 1] = 0
            vertex_indices[base + 2] = 0
            c0 = marching_cubes_clamp(scalar[i, j, k])
            c1 = marching_cubes_clamp(i < nx ? scalar[i + 1, j, k] : scalar[i, j, k])
            c2 = marching_cubes_clamp(j < ny ? scalar[i, j + 1, k] : scalar[i, j, k])
            c3 = marching_cubes_clamp(k < nz ? scalar[i, j, k + 1] : scalar[i, j, k])
            negative = c0 < 0
            if negative ? c1 > 0 : c1 < 0
                u = c0 / (c0 - c1)
                next += 1
                vertices[next] = SVector{3, Float32}(i - 1 + u, j - 1, k - 1)
                vertex_indices[base] = next
            end
            if negative ? c2 > 0 : c2 < 0
                u = c0 / (c0 - c2)
                next += 1
                vertices[next] = SVector{3, Float32}(i - 1, j - 1 + u, k - 1)
                vertex_indices[base + 1] = next
            end
            if negative ? c3 > 0 : c3 < 0
                u = c0 / (c0 - c3)
                next += 1
                vertices[next] = SVector{3, Float32}(i - 1, j - 1, k - 1 + u)
                vertex_indices[base + 2] = next
            end
        end
    end

    # Denormalize like `MarchingCubes.denormalize`
    @threaded backend for index in eachindex(vertices)
        @inbounds vertices[index] = vertex_offset .+ vertices[index] .* vertex_scale
    end

    # Pass 3: classify every cell of the box and count its triangles per slab. Cells in
    # the upper expansion layer have no positive corner and are skipped.
    cell_i = first(irange):(last(irange) - 1)
    cell_j = first(jrange):(last(jrange) - 1)
    cell_k = first(krange):(last(krange) - 1)
    cx, cy, cz = length(cell_i), length(cell_j), length(cell_k)
    n_cells = cx * cy * cz
    length(workspace.cell_cases) < n_cells && resize!(workspace.cell_cases, n_cells)
    cell_cases = workspace.cell_cases
    @inline cell_slot(i, j, k) = 1 + (i - i0) + cx * ((j - j0) + cy * (k - k0))

    triangle_counts = zeros(Int, cz)
    @threaded backend for k in cell_k
        count = 0
        @inbounds for j in cell_j, i in cell_i
            case = 0
            for p in 0:7
                value = scalar[i + ((p ⊻ (p >> 1)) & 1), j + ((p >> 1) & 1),
                               k + ((p >> 2) & 1)]
                marching_cubes_clamp(value) > 0 && (case += 1 << p)
            end
            cell_cases[cell_slot(i, j, k)] = case
            count += marching_cubes_triangle_count(case)
        end
        triangle_counts[k - k0 + 1] = count
    end
    triangle_starts = cumsum(triangle_counts) .- triangle_counts
    faces = Vector{Face}(undef, sum(triangle_counts))

    # Pass 4: emit triangles at their serial positions (edge codes as in `add_triangle`)
    @threaded backend for k in cell_k
        next = triangle_starts[k - k0 + 1]
        @inbounds for j in cell_j, i in cell_i
            case = cell_cases[cell_slot(i, j, k)]
            edges = MarchingCubes.casesClassic[case + 1]
            n_triangles = marching_cubes_triangle_count(case)
            for triangle in 1:n_triangles
                a = marching_cubes_edge_vertex(vertex_indices, slot, edges[3triangle - 2],
                                               i, j, k)
                b = marching_cubes_edge_vertex(vertex_indices, slot, edges[3triangle - 1],
                                               i, j, k)
                c = marching_cubes_edge_vertex(vertex_indices, slot, edges[3triangle],
                                               i, j, k)
                (a == 0 || b == 0 || c == 0) &&
                    error("marching cubes referenced an edge without a vertex")
                next += 1
                faces[next] = Face(a, b, c)
            end
        end
    end

    return SurfaceMesh(vertices, faces)
end

# Superset of the marching-cubes vertices that can have another vertex within
# `tolerance`: every vertex lies on a grid edge, and two edge vertices within `tolerance`
# are both within `tolerance` (plus rounding) of a common grid node — perpendicular edges
# give `d^2 = d_a^2 + d_b^2`, collinear ones `d = d_a + d_b`, and all other edge pairs are
# at least one voxel apart. A vertex is a candidate if its grid-index coordinates are
# within `threshold` of integers in all axes, where `threshold` bounds the tolerance plus
# the Float32 rounding of the index-space vertex and of its denormalization.
function grid_node_candidates(mesh, workspace, tolerance)
    offset = SVector{3, Float64}(workspace.vertex_offset)
    scale = SVector{3, Float64}(workspace.vertex_scale)
    dimensions = size(workspace.scratch)
    max_coordinate = maximum(abs.(offset) .+ abs.(scale) .* (dimensions .- 1))
    min_scale = minimum(scale)
    index_rounding = (maximum(dimensions) + 2 * max_coordinate / min_scale) * eps(Float32)
    threshold = tolerance / min_scale + 4 * index_rounding

    candidates = Vector{Bool}(undef, length(mesh.vertices))
    @threaded workspace.backend for index in eachindex(mesh.vertices)
        @inbounds begin
            coordinates = (SVector{3, Float64}(mesh.vertices[index]) - offset) ./ scale
            candidates[index] = all(abs.(coordinates .- round.(coordinates)) .<=
                                    threshold)
        end
    end
    return candidates
end

@inline function marching_cubes_triangle_count(case)
    edges = @inbounds MarchingCubes.casesClassic[case + 1]
    count = 0
    @inbounds while count < 5 && edges[3count + 1] > 0
        count += 1
    end
    return count
end

# Vertex id of edge `code` (1–12) of cell (i, j, k), mapping as in `add_triangle`
@inline function marching_cubes_edge_vertex(vertex_indices, slot, code, i, j, k)
    @inbounds begin
        code == 1 && return vertex_indices[slot(1, i, j, k)]
        code == 2 && return vertex_indices[slot(2, i + 1, j, k)]
        code == 3 && return vertex_indices[slot(1, i, j + 1, k)]
        code == 4 && return vertex_indices[slot(2, i, j, k)]
        code == 5 && return vertex_indices[slot(1, i, j, k + 1)]
        code == 6 && return vertex_indices[slot(2, i + 1, j, k + 1)]
        code == 7 && return vertex_indices[slot(1, i, j + 1, k + 1)]
        code == 8 && return vertex_indices[slot(2, i, j, k + 1)]
        code == 9 && return vertex_indices[slot(3, i, j, k)]
        code == 10 && return vertex_indices[slot(3, i + 1, j, k)]
        code == 11 && return vertex_indices[slot(3, i + 1, j + 1, k)]
        code == 12 && return vertex_indices[slot(3, i, j + 1, k)]
    end
    return Int32(0)
end

function evaluate_isovalue!(workspace, isovalue, target_volume; allow_empty=false,
                            region=axes(workspace.scratch))
    start_time = time_ns()
    mesh = contour!(workspace, isovalue; allow_empty, region)
    if isnothing(mesh)
        seconds = (time_ns() - start_time) / 1.0e9
        evaluation = Dict(
            "isovalue" => Float64(isovalue),
            "target_volume" => target_volume,
            "volume" => 0.0,
            "volume_error_percent" => -100.0,
            "seconds" => seconds,
            "vertices" => 0,
            "triangles" => 0,
            "merged_vertices" => 0,
            "rejected_nonmanifold_merges" => 0,
            "collapsed_triangles" => 0,
            "zero_volume_shells_discarded" => 0,
            "zero_volume_shell_vertices_discarded" => 0,
            "zero_volume_shell_triangles_discarded" => 0,
            "empty_mesh" => true
        )
        return nothing, -100.0, evaluation, nothing
    end
    candidates = grid_node_candidates(mesh, workspace, MESH_CLEANUP_TOLERANCE_M)
    mesh, cleanup = clean_near_duplicate_vertices(mesh; candidates)
    mesh, analysis,
    zero_volume_cleanup = prune_zero_volume_shells(mesh; backend=workspace.backend)
    component_volumes = isnothing(analysis) ? Float64[] : analysis.liquid_component_volumes
    volume = sum(component_volumes)
    error_percent = 100 * (volume - target_volume) / target_volume
    seconds = (time_ns() - start_time) / 1.0e9
    evaluation = Dict(
        "isovalue" => Float64(isovalue),
        "target_volume" => target_volume,
        "volume" => volume,
        "volume_error_percent" => error_percent,
        "seconds" => seconds,
        "vertices" => length(mesh.vertices),
        "triangles" => length(mesh.faces),
        "merged_vertices" => cleanup.merged_vertices,
        "rejected_nonmanifold_merges" => cleanup.rejected_nonmanifold_merges,
        "collapsed_triangles" => cleanup.collapsed_faces,
        "zero_volume_shells_discarded" => zero_volume_cleanup.removed_components,
        "zero_volume_shell_vertices_discarded" => zero_volume_cleanup.removed_vertices,
        "zero_volume_shell_triangles_discarded" => zero_volume_cleanup.removed_faces,
        "empty_mesh" => isnothing(analysis)
    )
    # The analysis is paired with the returned mesh object, which `prune_zero_volume_shells`
    # leaves unchanged when there is nothing to discard
    return isnothing(analysis) ? nothing : mesh, error_percent, evaluation, analysis
end

function corrected_contour!(workspace, base_isovalue, target_volume, options;
                            initial_isovalue=base_isovalue,
                            region=axes(workspace.scratch))
    evaluations = Dict{String, Any}[]
    allow_empty = true
    mesh, error_percent,
    evaluation,
    analysis = evaluate_isovalue!(workspace, initial_isovalue, target_volume;
                                  allow_empty, region)
    push!(evaluations, evaluation)
    best_mesh = mesh
    best_analysis = analysis
    best_error = isnothing(mesh) ? Inf : error_percent
    best_isovalue = initial_isovalue

    if abs(error_percent) <= options.volume_tolerance_percent
        isnothing(best_mesh) &&
            error("surface reconstruction produced an empty mesh")
        return best_mesh, best_isovalue, evaluations, best_analysis
    end

    if error_percent > 0
        lower_isovalue, lower_error = initial_isovalue, error_percent
        upper_isovalue = options.maximum_isovalue
        mesh, upper_error,
        evaluation,
        analysis = evaluate_isovalue!(workspace, upper_isovalue, target_volume;
                                      allow_empty=true, region)
        push!(evaluations, evaluation)
        upper_error <= 0 ||
            error("maximum isovalue does not bracket target volume")
    else
        upper_isovalue, upper_error = initial_isovalue, error_percent
        lower_isovalue = options.minimum_isovalue
        mesh, lower_error,
        evaluation,
        analysis = evaluate_isovalue!(workspace, lower_isovalue, target_volume;
                                      allow_empty=true, region)
        push!(evaluations, evaluation)
        lower_error >= 0 ||
            error("minimum isovalue does not bracket target volume")
    end
    if !isnothing(mesh) && abs(evaluation["volume_error_percent"]) < abs(best_error)
        best_mesh = mesh
        best_analysis = analysis
        best_error = evaluation["volume_error_percent"]
        best_isovalue = evaluation["isovalue"]
    end

    for _ in 1:options.volume_max_iterations
        width = upper_isovalue - lower_isovalue
        candidate_isovalue = (lower_isovalue * (-upper_error) +
                              upper_isovalue * lower_error) /
                             (lower_error - upper_error)
        # Safeguard regula falsi against a nearly stationary endpoint.
        candidate_isovalue = clamp(candidate_isovalue,
                                   lower_isovalue + 0.1width,
                                   upper_isovalue - 0.1width)
        mesh, candidate_error,
        evaluation,
        analysis = evaluate_isovalue!(workspace, candidate_isovalue,
                                      target_volume;
                                      allow_empty=true, region)
        push!(evaluations, evaluation)
        if !isnothing(mesh) && abs(candidate_error) < abs(best_error)
            best_mesh = mesh
            best_analysis = analysis
            best_error = candidate_error
            best_isovalue = candidate_isovalue
        end
        abs(candidate_error) <= options.volume_tolerance_percent && break
        if candidate_error > 0
            lower_isovalue, lower_error = candidate_isovalue, candidate_error
        else
            upper_isovalue, upper_error = candidate_isovalue, candidate_error
        end
    end
    isnothing(best_mesh) &&
        error("isovalue correction produced no nonempty mesh")
    abs(best_error) <= options.volume_tolerance_percent ||
        error("isovalue correction did not reach its volume tolerance")
    return best_mesh, best_isovalue, evaluations, best_analysis
end

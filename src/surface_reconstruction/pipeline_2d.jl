# Planar CIC reconstruction. Shared configuration and safeguarded correction are used
# in both dimensions; area replaces volume and marching squares replaces marching cubes.
# Every incident cell resolves an exact zero identically from its grid-node neighbors.
# Use a relative perturbation (not an absolute scalar epsilon) to keep near-node edges
# distinct in Float64 without moving them appreciably when the field is rescaled.
@inline function square_value(field, i, j)
    value = Float64(field[i, j])
    !iszero(value) && return value
    scale = Inf
    for (x, y) in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
        if 1 <= x <= size(field, 1) && 1 <= y <= size(field, 2)
            neighbor = abs(Float64(field[x, y]))
            !iszero(neighbor) && (scale = min(scale, neighbor))
        end
    end
    return sqrt(eps(Float64)) * (isfinite(scale) ? scale : 1.0)
end

function marching_squares(field, origin, spacing)
    nx, ny = size(field)
    vertices = SVector{2, Float64}[]
    horizontal, vertical = zeros(Int32, nx - 1, ny), zeros(Int32, nx, ny - 1)
    for j in 1:ny, i in 1:(nx - 1)
        a, b = square_value(field, i, j), square_value(field, i + 1, j)
        if (a > 0) != (b > 0)
            fraction = a / (a - b)
            push!(vertices, origin + spacing * SVector(i - 1 + fraction, j - 1))
            horizontal[i, j] = length(vertices)
        end
    end
    for j in 1:(ny - 1), i in 1:nx
        a, b = square_value(field, i, j), square_value(field, i, j + 1)
        if (a > 0) != (b > 0)
            fraction = a / (a - b)
            push!(vertices, origin + spacing * SVector(i - 1, j - 1 + fraction))
            vertical[i, j] = length(vertices)
        end
    end
    edges = ContourSegment[]
    for j in 1:(ny - 1), i in 1:(nx - 1)
        ids = (horizontal[i, j], vertical[i + 1, j], horizontal[i, j + 1], vertical[i, j])
        crossings = count(!iszero, ids)
        if crossings == 2
            endpoints = filter(!iszero, ids)
            push!(edges, ContourSegment(endpoints...))
        elseif crossings == 4
            a, b = square_value(field, i, j), square_value(field, i + 1, j)
            c, d = square_value(field, i + 1, j + 1), square_value(field, i, j + 1)
            # Asymptotic decider for bilinear saddle cells, evaluated in Float64.
            # No absolute epsilon on products: rescaling nonzero samples preserves
            # the decision. A zero determinant is a topological tie; choose one pairing.
            pairs = a * c - b * d >= 0 ? ((1, 2), (3, 4)) : ((1, 4), (2, 3))
            for (first, second) in pairs
                push!(edges, ContourSegment(ids[first], ids[second]))
            end
        elseif crossings != 0
            error("inconsistent marching-squares cell")
        end
    end
    return SurfaceMesh(vertices, edges)
end

function contour!(workspace::ReconstructionWorkspace2D, isovalue; allow_empty=false,
                  region=axes(workspace.scratch))
    (; field, constraint, scratch, backend, origin, spacing) = workspace
    @threaded backend for j in axes(field, 2)
        for i in axes(field, 1)
            scratch[i, j] = min(field[i, j] - Float32(isovalue), constraint[i, j])
        end
    end
    if !any(>(0), scratch)
        allow_empty && return nothing
        error("isovalue does not produce a nonempty contour")
    end
    all(<(0), scratch[1, :]) && all(<(0), scratch[end, :]) &&
    all(<(0), scratch[:, 1]) && all(<(0), scratch[:, end]) ||
        error("planar contour reaches the grid border; enlarge the reconstruction domain")
    mesh = marching_squares(scratch, origin, spacing)
    isempty(mesh.faces) && error("marching squares produced no segments")
    return mesh
end

function evaluate_isovalue!(workspace::ReconstructionWorkspace2D, isovalue, target_volume;
                            allow_empty=false, region=axes(workspace.scratch))
    start = time_ns()
    mesh = contour!(workspace, isovalue; allow_empty, region)
    analysis = isnothing(mesh) ? nothing : contour_analysis(mesh; check_intersections=false)
    area = isnothing(analysis) ? 0.0 : sum(analysis.liquid_component_volumes)
    error_percent = 100 * (area - target_volume) / target_volume
    evaluation = Dict{String, Any}("isovalue" => Float64(isovalue),
                                   "target_volume" => target_volume,
                                   "volume" => area, "area" => area,
                                   "volume_error_percent" => error_percent,
                                   "seconds" => (time_ns() - start) / 1.0e9,
                                   "vertices" => isnothing(mesh) ? 0 :
                                                 length(mesh.vertices),
                                   "segments" => isnothing(mesh) ? 0 : length(mesh.faces),
                                   "empty_mesh" => isnothing(mesh))
    return mesh, error_percent, evaluation, analysis
end

function equivalent_area_circle(center, area; n_segments=64)
    # Use polygon area, not pi*r², so the returned polygon itself meets the target.
    radius = sqrt(2area / (n_segments * sin(2pi / n_segments)))
    vertices = [center +
                radius * SVector(cospi(2index / n_segments), sinpi(2index / n_segments))
                for index in 0:(n_segments - 1)]
    faces = [ContourSegment(index, mod1(index + 1, n_segments)) for index in 1:n_segments]
    return SurfaceMesh(vertices, faces), radius
end

function unresolved_planar_components(workspace, points, areas, components, excluded,
                                      isovalue, domain, boundaries, clearance)
    (; field, constraint, origin, spacing) = workspace
    fallbacks = NamedTuple[]
    for (index, particles) in enumerate(components)
        index in excluded && continue
        area = sum(areas[particles])
        center = sum(areas[p] * SVector{2, Float64}(points[:, p]) for p in particles) / area
        mesh, radius = equivalent_area_circle(center, area)
        lower = min.(center .- radius,
                     SVector(minimum(points[1, particles]), minimum(points[2, particles]))) .-
                spacing
        upper = max.(center .+ radius,
                     SVector(maximum(points[1, particles]), maximum(points[2, particles]))) .+
                spacing
        first_node = max.(floor.(Int, (lower - origin) / spacing) .+ 1, 1)
        last_node = min.(ceil.(Int, (upper - origin) / spacing) .+ 1, SVector(size(field)))
        any(field[i, j] > isovalue && constraint[i, j] > 0
            for j in first_node[2]:last_node[2], i in first_node[1]:last_node[1]) &&
            continue
        length(particles) <= SPARSE_COMPONENT_MAXIMUM_PARTICLES ||
            error("unresolved planar component exceeds the sparse-fallback particle limit")
        maximum(norm(SVector{2, Float64}(points[:, p]) - center) for p in particles) <=
        radius ||
            error("unresolved planar component is not compact enough for a circular fallback")
        domain_distance(domain, center...) > radius &&
        all(boundary -> signed_distance(boundary.bvh, center) > radius + clearance,
            boundaries) ||
            error("area-equivalent circle crosses a solid constraint")
        push!(fallbacks,
              (; component_index=index, particles, source_volume=area, center, radius,
               mesh))
    end
    return fallbacks
end

function _reconstruct!(workspace::ReconstructionWorkspace2D, points, volumes, boundaries,
                       origin, spacing, domain::ReconstructionDomain{2}, particle_spacing,
                       sigma_voxels, boundary_clearance, support, options;
                       initial_isovalue=options.isovalue)
    timings = Dict{String, Float64}()
    history = options.record_field_moments ? Dict{String, Any}[] : nothing
    start = time_ns()
    distance_scale = max(sigma_voxels * spacing, spacing)
    domain_constraint!(workspace.constraint, origin, spacing, domain, distance_scale;
                       backend=workspace.backend)
    n_samples = add_boundary_constraints!(workspace.constraint, boundaries, origin, spacing,
                                          boundary_clearance, distance_scale;
                                          backend=workspace.backend)
    timings["constraints"] = (time_ns() - start) / 1.0e9
    start = time_ns()
    deposit = deposit_volume_cic!(workspace.field, points, volumes, origin, spacing)
    isnothing(history) ||
        record_field_moments!(history, "cic_deposition", workspace.field, origin, spacing)
    timings["cic_deposition"] = (time_ns() - start) / 1.0e9
    start = time_ns()
    gaussian_filter!(workspace.field, workspace.temporary, workspace.scratch,
                     gaussian_kernel(sigma_voxels); backend=workspace.backend)
    isnothing(history) ||
        record_field_moments!(history, "gaussian_filter", workspace.field, origin, spacing)
    timings["gaussian_filter"] = (time_ns() - start) / 1.0e9
    target = sum(volumes)
    isfinite(target) && target > 0 ||
        throw(ArgumentError("total particle area must be finite and positive"))
    start = time_ns()
    mesh, level, evaluations,
    analysis = corrected_contour!(workspace, options.isovalue, target,
                                  options; initial_isovalue)
    fallbacks = NamedTuple[]
    primary_target = target
    if options.sparse_component_fallback
        components = particle_components(points,
                                         SPARSE_COMPONENT_GRAPH_RADIUS_PER_SPACING *
                                         particle_spacing)
        excluded = Set{Int}()
        for iteration in 1:SPARSE_FALLBACK_MAXIMUM_ITERATIONS
            additions = unresolved_planar_components(workspace, points, volumes, components,
                                                     excluded, level, domain, boundaries,
                                                     boundary_clearance)
            isempty(additions) && break
            append!(fallbacks, additions)
            validate_disjoint_fallbacks(fallbacks)
            union!(excluded, (fallback.component_index for fallback in additions))
            primary_target = target - sum(fallback.source_volume for fallback in fallbacks)
            primary_target > 0 ||
                error("sparse circles leave no primary reconstruction area")
            mesh, level, local_evaluations,
            analysis = corrected_contour!(workspace, options.isovalue,
                                          primary_target, options;
                                          initial_isovalue=level)
            append!(evaluations, local_evaluations)
        end
        isempty(unresolved_planar_components(workspace, points, volumes, components,
                                             excluded,
                                             level, domain, boundaries, boundary_clearance)) ||
            error("planar sparse fallback did not stabilize")
        if !isempty(fallbacks)
            mesh = combine_surface_meshes(vcat([mesh],
                                               [fallback.mesh for fallback in fallbacks]))
            analysis = contour_analysis(mesh)
        end
    end
    timings["contouring_and_volume_correction"] = (time_ns() - start) / 1.0e9
    start = time_ns()
    mesh = SurfaceMesh(mesh.vertices, analysis.oriented_faces)
    geometry = mesh_geometry_stats(mesh; analysis)
    validate_reconstructed_geometry(geometry, target, options.volume_tolerance_percent)
    n_inside = count(vertex -> any(boundary -> signed_distance(boundary.bvh, vertex) <
                                               -1.0e-7,
                                   boundaries), mesh.vertices)
    timings["final_geometry_statistics"] = (time_ns() - start) / 1.0e9
    timings["total"] = sum(values(timings))
    field_min, field_max = extrema(workspace.field)
    field_integral = sum(Float64, workspace.field) * spacing^2
    details = Dict{String, Any}("ndims" => 2, "area" => geometry.volume,
                                "perimeter" => geometry.surface_area,
                                "particle_area" => target,
                                "particle_volume" => target,
                                "deposited_volume" => deposit.deposited_volume,
                                "grid_dimensions" => collect(size(workspace.field)),
                                "particles_within_grid" => size(points, 2),
                                "field_min" => Float64(field_min),
                                "field_max" => Float64(field_max),
                                "field_integral" => field_integral,
                                "filtered_field_integral" => field_integral,
                                "field_deposition_stats" => Dict{String, Any}(
                                    "deposited_volume" => deposit.deposited_volume,
                                    "particle_volume" => target
                                ),
                                "boundary_sample_grid_points" => n_samples,
                                "base_isovalue" => options.isovalue,
                                "initial_isovalue" => initial_isovalue,
                                "effective_isovalue" => level,
                                "isovalue_correction_evaluations" => evaluations,
                                "n_vertices_inside_boundaries" => n_inside,
                                "n_degenerate_segments" => 0,
                                "sparse_component_fallback" => options.sparse_component_fallback,
                                "n_sparse_fallback_components" => length(fallbacks),
                                "n_sparse_fallback_particles" => sum(length(f.particles)
                                                                     for f in fallbacks;
                                                                     init=0),
                                "sparse_fallback_volume" => sum(f.source_volume
                                                                for f in fallbacks;
                                                                init=0.0),
                                "sparse_primary_target_volume" => primary_target,
                                "timings" => timings)
    for (name, value) in pairs(geometry)
        details[string(name)] = value
    end
    details["bounds"] = [
        geometry.lower[1],
        geometry.upper[1],
        geometry.lower[2],
        geometry.upper[2]
    ]
    violations = [domain_violation(domain, vertex) for vertex in mesh.vertices]
    details["closed_boundary_vertices_outside_implicit_domain"] = count(>(1.0e-7),
                                                                        violations)
    details["maximum_closed_boundary_violation"] = maximum(violations)
    if !isnothing(history)
        record_field_moments!(history, "final_field", workspace.field, origin, spacing)
        details["field_deposition_stats"]["field_moment_history"] = history
    end
    return mesh, SurfaceReconstructionStatistics(details), analysis
end

# Reconstruction pipeline: deposit, filter, constrain, correct, extract. `support` is the
# deposition support of `points` (see `deposition_support`), precomputed together with
# the outside-grid exclusion by `exclude_and_support`.
function _reconstruct!(workspace, points, volumes, boundaries, origin, spacing,
                       domain::ReconstructionDomain, particle_spacing, sigma_voxels,
                       boundary_clearance, support,
                       options; initial_isovalue=options.isovalue)
    timings = Dict{String, Float64}()
    field_moment_history = options.record_field_moments ? Dict{String, Any}[] : nothing
    start_time = time_ns()
    distance_scale = max(sigma_voxels * spacing, spacing)
    # The domain constraint only depends on the grid and the configuration. A reused
    # workspace already holds it, except where the last boundary constraints modified it.
    constraint_key = (domain, SVector{3, Float64}(origin), Float64(spacing),
                      Float64(distance_scale))
    if workspace.constraint_key == constraint_key
        for region in workspace.boundary_regions
            domain_constraint!(workspace.constraint, origin, spacing, domain,
                               distance_scale; backend=workspace.backend, region)
        end
    else
        domain_constraint!(workspace.constraint, origin, spacing, domain, distance_scale;
                           backend=workspace.backend)
    end
    # Invalid until the boundary constraints are complete
    workspace.constraint_key = nothing
    boundary_regions = boundary_constraint_regions(boundaries, origin, spacing,
                                                   boundary_clearance,
                                                   size(workspace.constraint))
    boundary_sample_count = add_boundary_constraints!(workspace.constraint, boundaries,
                                                      origin,
                                                      spacing, boundary_clearance,
                                                      distance_scale;
                                                      backend=workspace.backend)
    workspace.boundary_regions = NTuple{3, UnitRange{Int}}[region
                                                           for region in boundary_regions]
    workspace.constraint_key = constraint_key
    timings["constraints"] = (time_ns() - start_time) / 1.0e9

    start_time = time_ns()
    (; field, temporary, scratch) = workspace
    dimensions = size(field)
    kernel = gaussian_kernel(sigma_voxels)
    # Grid regions touched by this reconstruction: the deposition writes the particle
    # stencils (`support`), the filtered field is nonzero only within `field_region`, and
    # marching cubes reads the contouring scalar only within `contour_region` (see
    # `contour!`), since for isovalues `>= eps(Float32)`, voxels outside `field_region` are
    # negative. All evaluated isovalues are `>= min(initial_isovalue, minimum_isovalue)`.
    field_region = expand_region(support, length(kernel) >>> 1, dimensions)
    contour_region = min(initial_isovalue, options.minimum_isovalue) >= eps(Float32) ?
                     expand_region(field_region, 2, dimensions) : full_region(dimensions)
    # Instead of zeroing the full grid, zero the region written by the last reconstruction
    # of this workspace and record the region that this one writes, so that all buffers
    # are zero outside `contour_region` whenever this function exits.
    for buffer in (field, temporary, scratch)
        parallel_fill!(buffer, 0.0f0; backend=workspace.backend,
                       region=workspace.dirty_region)
    end
    workspace.dirty_region = contour_region

    deposition_stats = deposit_volume_cic!(field, points, volumes, origin, spacing;
                                           backend=workspace.backend, support)
    isnothing(field_moment_history) ||
        record_field_moments!(field_moment_history, "cic_deposition", field,
                              origin, spacing)
    timings["cic_deposition"] = (time_ns() - start_time) / 1.0e9

    start_time = time_ns()
    gaussian_filter!(field, temporary, scratch, kernel; backend=workspace.backend,
                     support, zeroed=true)
    isnothing(field_moment_history) ||
        record_field_moments!(field_moment_history, "gaussian_filter", field,
                              origin, spacing)
    timings["gaussian_filter"] = (time_ns() - start_time) / 1.0e9
    start_time = time_ns()
    target_volume = sum(volumes)
    isfinite(target_volume) && target_volume > 0 ||
        throw(ArgumentError("total particle volume must be finite and positive"))
    n_sparse_source_components = nothing
    sparse_fallback_iterations = 0
    sparse_fallbacks = NamedTuple[]
    primary_target_volume = target_volume
    mesh, effective_isovalue,
    evaluations,
    best_analysis = corrected_contour!(workspace, options.isovalue,
                                       target_volume, options;
                                       initial_isovalue=initial_isovalue,
                                       region=contour_region)
    for evaluation in evaluations
        evaluation["sparse_fallback_iteration"] = 0
    end
    if options.sparse_component_fallback
        graph_radius = SPARSE_COMPONENT_GRAPH_RADIUS_PER_SPACING * particle_spacing
        components = particle_components(points, graph_radius)
        n_sparse_source_components = length(components)
        excluded_components = Set{Int}()
        stable = false
        for iteration in 1:SPARSE_FALLBACK_MAXIMUM_ITERATIONS
            new_fallbacks = unresolved_sparse_components(workspace.field,
                                                         workspace.constraint, points,
                                                         volumes, components,
                                                         excluded_components, origin,
                                                         spacing, effective_isovalue)
            if isempty(new_fallbacks)
                stable = true
                break
            end
            append!(sparse_fallbacks, new_fallbacks)
            validate_disjoint_fallbacks(sparse_fallbacks)
            union!(excluded_components,
                   (fallback.component_index for fallback in new_fallbacks))
            sparse_fallback_iterations = iteration
            fallback_volume = sum(fallback.source_volume
                                  for fallback in sparse_fallbacks)
            primary_target_volume = target_volume - fallback_volume
            primary_target_volume > 0 ||
                error("sparse fallback volume leaves no positive primary reconstruction target")
            mesh, effective_isovalue,
            local_evaluations,
            best_analysis = corrected_contour!(workspace,
                                               options.isovalue,
                                               primary_target_volume,
                                               options;
                                               initial_isovalue=effective_isovalue,
                                               region=contour_region)
            for evaluation in local_evaluations
                evaluation["sparse_fallback_iteration"] = iteration
            end
            append!(evaluations, local_evaluations)
        end
        if !stable
            remaining = unresolved_sparse_components(workspace.field, workspace.constraint,
                                                     points, volumes, components,
                                                     excluded_components, origin, spacing,
                                                     effective_isovalue)
            isempty(remaining) ||
                error("sparse-component fallback did not stabilize after " *
                      "$SPARSE_FALLBACK_MAXIMUM_ITERATIONS iterations")
        end
        if !isempty(sparse_fallbacks)
            meshes = SurfaceMesh[mesh]
            append!(meshes, (fallback.mesh for fallback in sparse_fallbacks))
            mesh = combine_surface_meshes(meshes)
        end
    end
    timings["contouring_and_volume_correction"] = (time_ns() - start_time) / 1.0e9
    field_sum, field_min,
    field_max = field_sum_extrema(field; backend=workspace.backend, region=field_region)
    field_integral = field_sum * spacing^3
    if !isnothing(field_moment_history)
        record_field_moments!(field_moment_history, "final_field", workspace.field,
                              origin, spacing)
        deposition_stats = merge(deposition_stats,
                                 (field_moment_history=field_moment_history,))
    end

    start_time = time_ns()
    mesh_before_collapse = mesh
    mesh, degenerate_cleanup = collapse_degenerate_triangles(mesh)
    timings["final_degenerate_triangle_cleanup"] = (time_ns() - start_time) / 1.0e9

    start_time = time_ns()
    # The winning correction evaluation already ran the liquid analysis on this exact
    # mesh object; reuse it when neither the sparse fallback (which combines meshes) nor
    # the degenerate cleanup changed the mesh
    reusable_analysis = isempty(sparse_fallbacks) &&
                        degenerate_cleanup.collapsed_edges == 0 &&
                        mesh === mesh_before_collapse && !isnothing(best_analysis)
    geometry = mesh_geometry_stats(mesh;
                                   analysis=reusable_analysis ? best_analysis : nothing,
                                   backend=workspace.backend)
    validate_reconstructed_geometry(geometry, target_volume,
                                    options.volume_tolerance_percent)
    inside_boundaries = vertices_inside_boundaries(mesh, boundaries;
                                                   backend=workspace.backend)
    timings["final_geometry_statistics"] = (time_ns() - start_time) / 1.0e9
    timings["total"] = sum(values(timings))

    # The mesh bounding box lies inside the domain iff all its corners do.
    violations = maximum(domain_violation(domain,
                                          SVector(corner_x, corner_y, corner_z))
                         for corner_x in (geometry.lower[1], geometry.upper[1]),
                             corner_y in (geometry.lower[2], geometry.upper[2]),
                             corner_z in (geometry.lower[3], geometry.upper[3]))
    outside = violations > 1.0e-7 ?
              count(vertex -> domain_violation(domain, SVector{3, Float64}(vertex)) >
                              1.0e-7, mesh.vertices) : 0
    final_evaluation_index = findlast(evaluations) do evaluation
        isapprox(evaluation["isovalue"], effective_isovalue; rtol=0, atol=1.0e-12) &&
            isapprox(evaluation["target_volume"], primary_target_volume;
                     rtol=0, atol=1.0e-12)
    end
    isnothing(final_evaluation_index) &&
        error("final contour evaluation is missing")
    final_evaluation = evaluations[final_evaluation_index]
    sparse_fallback_records = [Dict(
                                   "source_component_index" => fallback.component_index,
                                   "source_particle_indices" => Int.(fallback.particles),
                                   "n_particles" => length(fallback.particles),
                                   "source_volume" => fallback.source_volume,
                                   "polygon_volume" => abs(mesh_signed_volume(fallback.mesh)),
                                   "center" => collect(fallback.center),
                                   "polygon_radius" => fallback.radius,
                                   "maximum_particle_distance_from_center" => fallback.maximum_particle_distance,
                                   "minimum_sampled_solid_constraint" => fallback.minimum_constraint
                               ) for fallback in sparse_fallbacks]

    stats = Dict{String, Any}(
        "grid_dimensions" => collect(size(workspace.field)),
        "particles_within_grid" => size(points, 2),
        "field_min" => Float64(field_min),
        "field_max" => Float64(field_max),
        "field_integral" => field_integral,
        "filtered_field_integral" => field_integral,
        "field_deposition_stats" => Dict(string(key) => value
                                         for (key, value) in pairs(deposition_stats)),
        "boundary_sample_grid_points" => boundary_sample_count,
        "base_isovalue" => options.isovalue,
        "initial_isovalue" => initial_isovalue,
        "effective_isovalue" => effective_isovalue,
        "isovalue_correction_evaluations" => evaluations,
        "mesh_cleanup" => "merge vertices within absolute tolerance only when the merged vertex link remains one closed manifold cycle, discard collapsed triangles and connected shells with signed volume at floating-point zero, then collapse degenerate final-mesh edges only when topology remains manifold",
        "mesh_cleanup_tolerance" => MESH_CLEANUP_TOLERANCE_M,
        "mesh_vertices_merged" => final_evaluation["merged_vertices"],
        "mesh_vertex_merges_rejected_for_topology" => final_evaluation["rejected_nonmanifold_merges"],
        "mesh_collapsed_triangles_discarded" => final_evaluation["collapsed_triangles"],
        "mesh_zero_volume_shells_discarded" => final_evaluation["zero_volume_shells_discarded"],
        "mesh_zero_volume_shell_vertices_discarded" => final_evaluation["zero_volume_shell_vertices_discarded"],
        "mesh_zero_volume_shell_triangles_discarded" => final_evaluation["zero_volume_shell_triangles_discarded"],
        "mesh_degenerate_edges_collapsed" => degenerate_cleanup.collapsed_edges,
        "mesh_degenerate_cleanup_vertices_discarded" => degenerate_cleanup.removed_vertices,
        "mesh_degenerate_cleanup_triangles_discarded" => degenerate_cleanup.removed_faces,
        "mesh_degenerate_cleanup_maximum_edge_length" => degenerate_cleanup.maximum_edge_length,
        "sparse_component_fallback" => options.sparse_component_fallback,
        "n_sparse_source_components" => n_sparse_source_components,
        "sparse_fallback_iterations" => sparse_fallback_iterations,
        "n_sparse_fallback_components" => length(sparse_fallbacks),
        "n_sparse_fallback_particles" => sum(length(fallback.particles)
                                             for fallback in sparse_fallbacks; init=0),
        "sparse_fallback_volume" => sum(fallback.source_volume
                                        for fallback in sparse_fallbacks; init=0.0),
        "sparse_primary_target_volume" => primary_target_volume,
        "sparse_fallback_components" => sparse_fallback_records,
        "deposited_volume" => deposition_stats.deposited_volume,
        "particle_volume" => deposition_stats.particle_volume,
        "closed_boundary_vertices_outside_implicit_domain" => outside,
        "maximum_closed_boundary_violation" => outside > 0 ? violations : 0.0,
        "closed_boundary_vertices_clipped_after_smoothing" => 0,
        "maximum_closed_boundary_clip_distance" => 0.0,
        "bounds" => [geometry.lower[1], geometry.upper[1],
            geometry.lower[2], geometry.upper[2],
            geometry.lower[3], geometry.upper[3]],
        "n_boundary_edges" => geometry.n_boundary_edges,
        "n_nonmanifold_edges" => geometry.n_nonmanifold_edges,
        "n_degenerate_triangles" => geometry.n_degenerate_triangles,
        "n_vertices_inside_boundaries" => inside_boundaries,
        "volume" => geometry.volume,
        "surface_area" => geometry.surface_area,
        "region_volumes" => geometry.region_volumes,
        "n_connected_regions" => geometry.n_connected_regions,
        "detached_region_volume" => geometry.detached_region_volume,
        "n_surface_components" => geometry.n_surface_components,
        "shell_volumes" => geometry.shell_volumes,
        "shell_signed_volumes" => geometry.shell_signed_volumes,
        "shell_nesting_depths" => geometry.shell_nesting_depths,
        "n_cavity_regions" => geometry.n_cavity_regions,
        "cavity_volume" => geometry.cavity_volume,
        "timings" => timings
    )
    # The reusable analysis describes the returned mesh object; writers
    # (`trixi2vtk`, `write_ply`) accept it to skip their own liquid analysis
    return mesh, SurfaceReconstructionStatistics(stats),
           reusable_analysis ? best_analysis : nothing
end

# The isovalue search certifies its primary mesh before the final cleanup and optional
# fallback combination. Certify the mesh actually returned as well: these later stages
# can change both its volume and topology.
function validate_reconstructed_geometry(geometry, target_volume, tolerance_percent)
    isfinite(geometry.volume) && geometry.volume > 0 ||
        error("final reconstructed mesh has nonpositive or non-finite volume")
    abs(100 * (geometry.volume - target_volume) / target_volume) <= tolerance_percent ||
        error("final reconstructed mesh does not satisfy the volume tolerance after cleanup")
    geometry.n_boundary_edges == 0 && geometry.n_nonmanifold_edges == 0 &&
    geometry.n_degenerate_triangles == 0 ||
        error("final reconstructed mesh has boundary, nonmanifold or degenerate triangles")
    return nothing
end

# Surface meshes: cleanup, liquid-domain analysis, and geometry statistics.
"""
    SurfaceMesh{ELTYPE, INTTYPE, NDIMS}

Free surface with `vertices` positions and 1-based `faces`: triangles in 3D and line
segments in 2D. 3D reconstruction produces Float32 vertices; planar contours retain
Float64 coordinates to resolve near-node intersections. Indices are Int32.
The first two type parameters retain their element/index-type meaning.
"""
struct SurfaceMesh{ELTYPE, INTTYPE, NDIMS}
    vertices::Vector{SVector{NDIMS, ELTYPE}}
    faces::Vector{SVector{NDIMS, INTTYPE}}
end

# Retain the two-parameter construction spelling while inferring geometric dimension.
function SurfaceMesh{T, I}(vertices::AbstractVector{<:SVector{N}},
                           faces::AbstractVector{<:SVector{N}}) where {T, I, N}
    return SurfaceMesh{T, I, N}(vertices, faces)
end

Base.eltype(::SurfaceMesh{ELTYPE}) where {ELTYPE} = ELTYPE
Base.ndims(::SurfaceMesh{T, I, N}) where {T, I, N} = N

function surface_mesh(mc)
    vertices = Vector{SVector{3, Float32}}(mc.vertices)
    faces = Vector{Face}(undef, length(mc.triangles))
    @inbounds for index in eachindex(mc.triangles)
        triangle = mc.triangles[index]
        faces[index] = Face(triangle[1], triangle[2], triangle[3])
    end
    return SurfaceMesh(vertices, faces)
end

function combine_surface_meshes(meshes)
    isempty(meshes) && throw(ArgumentError("cannot combine an empty mesh collection"))
    vertex_type = eltype(first(meshes).vertices)
    index_type = eltype(first(meshes).faces)
    all(mesh -> eltype(mesh.vertices) == vertex_type &&
                eltype(mesh.faces) == index_type, meshes) ||
        throw(ArgumentError("combined meshes must share vertex and index element types"))
    vertices = vertex_type[]
    faces = Vector{index_type}()
    sizehint!(vertices, sum(length(mesh.vertices) for mesh in meshes))
    sizehint!(faces, sum(length(mesh.faces) for mesh in meshes))
    for mesh in meshes
        offset = eltype(index_type)(length(vertices))
        append!(vertices, mesh.vertices)
        append!(faces,
                (face .+ offset
                 for face in mesh.faces))
    end
    return SurfaceMesh(vertices, faces)
end

"""
    TriangleMesh(mesh::SurfaceMesh)

Convert a reconstructed surface to a [`TriangleMesh`](@ref) for reuse with the
preprocessing geometry infrastructure (e.g. `SignedDistanceField`, `ComplexShape`,
`FaceNeighborhoodSearch`), for example to re-initialise a simulation from a
reconstructed free surface or to constrain a later frame by an earlier one. Vertex
positions convert exactly from `Float32`; face normals are recomputed from the cleaned
triangles.
"""
function TriangleMesh(mesh::SurfaceMesh{T, I, 3}) where {T, I}
    vertices = [SVector{3, Float64}(vertex) for vertex in mesh.vertices]
    parent, reverse_winding = reflected_component_winding(mesh)
    flags = vertex_reverse_flags(parent, reverse_winding, length(mesh.vertices))
    face_vertices = [flags[face[1]] ?
                     (vertices[face[1]], vertices[face[3]], vertices[face[2]]) :
                     (vertices[face[1]], vertices[face[2]], vertices[face[3]])
                     for face in mesh.faces]
    face_normals = map(face_vertices) do (a, b, c)
        normal = cross(b - a, c - a)
        magnitude = norm(normal)
        magnitude > eps(Float64) ||
            throw(ArgumentError("reconstructed surface contains a degenerate triangle"))
        normal / magnitude
    end
    return TriangleMesh(face_vertices, face_normals, vertices)
end

# Whether a face references three distinct vertices. Unlike `allunique`, this does not
# allocate for static vectors.
@inline function distinct_vertices(face)
    return face[1] != face[2] && face[2] != face[3] && face[1] != face[3]
end

function merged_vertex_link_is_manifold(mesh, remapping, output_vertex, face_indices)
    link_neighbors = Dict{Int32, Set{Int32}}()
    link_edges = Set{Tuple{Int32, Int32}}()
    face_keys = Set{NTuple{3, Int32}}()
    for face_index in face_indices
        face = mesh.faces[face_index]
        remapped = Face(remapping[face[1]], remapping[face[2]], remapping[face[3]])
        distinct_vertices(remapped) || continue
        output_vertex in remapped || continue

        ordered = sort(collect(remapped))
        face_key = (ordered[1], ordered[2], ordered[3])
        face_key in face_keys && return false
        push!(face_keys, face_key)

        opposite = Int32[vertex for vertex in remapped if vertex != output_vertex]
        length(opposite) == 2 || return false
        link_edge = minmax(opposite[1], opposite[2])
        link_edge in link_edges && return false
        push!(link_edges, link_edge)
        push!(get!(link_neighbors, opposite[1], Set{Int32}()), opposite[2])
        push!(get!(link_neighbors, opposite[2], Set{Int32}()), opposite[1])
    end
    isempty(link_neighbors) && return false
    all(length(neighbors) == 2 for neighbors in values(link_neighbors)) || return false

    first_vertex = first(keys(link_neighbors))
    visited = Set{Int32}((first_vertex,))
    stack = Int32[first_vertex]
    while !isempty(stack)
        vertex = pop!(stack)
        for neighbor in link_neighbors[vertex]
            neighbor in visited && continue
            push!(visited, neighbor)
            push!(stack, neighbor)
        end
    end
    return length(visited) == length(link_neighbors)
end

# `candidates` optionally marks a superset of the vertices that can have another vertex
# within `tolerance` (see `grid_node_candidates`). All other vertices are isolated: they
# never match and are never matched, so they are kept in order without entering the
# spatial hash. The result is identical to the unrestricted search.
function clean_near_duplicate_vertices(mesh; tolerance=MESH_CLEANUP_TOLERANCE_M,
                                       candidates=nothing)
    if candidates !== nothing && !any(candidates)
        # No vertex can merge, so the remapping is the identity. Faces that already
        # reference duplicate vertices would still collapse, so only skip the cleanup
        # when every face is intact (always true for marching-cubes output, whose edge
        # vertices have unique ids).
        if !any(face -> !distinct_vertices(face), mesh.faces)
            return mesh,
                   (merged_vertices=0, rejected_nonmanifold_merges=0,
                    collapsed_faces=0, input_vertices=length(mesh.vertices),
                    input_faces=length(mesh.faces))
        end
    end
    bucket_heads = Dict{NTuple{3, Int64}, Int32}()
    sizehint!(bucket_heads,
              isnothing(candidates) ? length(mesh.vertices) : count(candidates))
    next_in_bucket = zeros(Int32, length(mesh.vertices))
    remapping = Vector{Int32}(undef, length(mesh.vertices))
    vertices = SVector{3, Float32}[]
    representative_old_indices = Int32[]
    sizehint!(vertices, length(mesh.vertices))
    sizehint!(representative_old_indices, length(mesh.vertices))
    tolerance_squared = tolerance^2

    for (old_index, vertex) in enumerate(mesh.vertices)
        if candidates !== nothing && !candidates[old_index]
            push!(vertices, vertex)
            push!(representative_old_indices, Int32(old_index))
            remapping[old_index] = Int32(length(vertices))
            continue
        end
        point = SVector{3, Float64}(vertex)
        cell = ntuple(axis -> floor(Int64, point[axis] / tolerance), 3)
        match_index = Int32(0)
        for z_offset in -1:1, y_offset in -1:1, x_offset in -1:1
            key = (cell[1] + x_offset, cell[2] + y_offset, cell[3] + z_offset)
            candidate = get(bucket_heads, key, Int32(0))
            while candidate != 0
                delta = point - SVector{3, Float64}(vertices[candidate])
                if dot(delta, delta) <= tolerance_squared
                    match_index = candidate
                    break
                end
                candidate = next_in_bucket[candidate]
            end
            match_index != 0 && break
        end
        if match_index == 0
            push!(vertices, vertex)
            push!(representative_old_indices, Int32(old_index))
            match_index = Int32(length(vertices))
            next_in_bucket[match_index] = get(bucket_heads, cell, Int32(0))
            bucket_heads[cell] = match_index
        end
        remapping[old_index] = match_index
    end

    merge_groups = Dict{Int32, Vector{Int32}}()
    @inbounds for old_index in eachindex(remapping)
        output_index = remapping[old_index]
        representative = representative_old_indices[output_index]
        old_index == representative && continue
        if !haskey(merge_groups, output_index)
            merge_groups[output_index] = Int32[representative]
        end
        push!(merge_groups[output_index], Int32(old_index))
    end
    candidate_group_by_vertex = zeros(Int32, length(mesh.vertices))
    candidate_face_indices = Dict(group_index => Int32[]
                                  for group_index in keys(merge_groups))
    for (group_index, members) in merge_groups
        candidate_group_by_vertex[members] .= group_index
    end
    @inbounds for (face_index, face) in enumerate(mesh.faces)
        first_group = candidate_group_by_vertex[face[1]]
        second_group = candidate_group_by_vertex[face[2]]
        third_group = candidate_group_by_vertex[face[3]]
        first_group == 0 || push!(candidate_face_indices[first_group], Int32(face_index))
        (second_group == 0 || second_group == first_group) ||
            push!(candidate_face_indices[second_group], Int32(face_index))
        (third_group == 0 || third_group == first_group || third_group == second_group) ||
            push!(candidate_face_indices[third_group], Int32(face_index))
    end

    rejected_merges = 0
    rejected_groups = Int32[]
    for (group_index, members) in merge_groups
        merged_vertex_link_is_manifold(mesh, remapping, group_index,
                                       candidate_face_indices[group_index]) && continue
        push!(rejected_groups, group_index)
        rejected_merges += length(members) - 1
    end
    for group_index in rejected_groups
        representative = representative_old_indices[group_index]
        for old_index in merge_groups[group_index]
            old_index == representative && continue
            push!(vertices, mesh.vertices[old_index])
            remapping[old_index] = Int32(length(vertices))
        end
    end

    faces = Face[]
    sizehint!(faces, length(mesh.faces))
    collapsed_faces = 0
    @inbounds for face in mesh.faces
        remapped = Face(remapping[face[1]], remapping[face[2]], remapping[face[3]])
        if distinct_vertices(remapped)
            push!(faces, remapped)
        else
            collapsed_faces += 1
        end
    end
    cleaned = SurfaceMesh(vertices, faces)
    return cleaned,
           (merged_vertices=length(mesh.vertices) - length(vertices),
            rejected_nonmanifold_merges=rejected_merges,
            collapsed_faces=collapsed_faces,
            input_vertices=length(mesh.vertices),
            input_faces=length(mesh.faces))
end

@inline function find_root!(parent, vertex::Integer)
    index = eltype(parent)(vertex)
    @inbounds while parent[index] != index
        parent[index] = parent[parent[index]]
        index = parent[index]
    end
    return index
end

@inline function union_vertices!(parent, sizes, first::Integer, second::Integer)
    first_root = find_root!(parent, first)
    second_root = find_root!(parent, second)
    first_root == second_root && return first_root
    @inbounds if sizes[first_root] < sizes[second_root]
        first_root, second_root = second_root, first_root
    end
    @inbounds begin
        parent[second_root] = first_root
        sizes[first_root] += sizes[second_root]
    end
    return first_root
end

struct MeshShell
    root::Int32
    face_indices::Vector{Int32}
    signed_volume::Float64
    volume::Float64
    lower::SVector{3, Float64}
    upper::SVector{3, Float64}
    probe::SVector{3, Float64}
end

struct MeshLiquidAnalysis
    vertex_parent::Vector{Int32}
    shells::Vector{MeshShell}
    shell_by_root::Dict{Int32, Int32}
    shell_parents::Vector{Int32}
    shell_depths::Vector{Int32}
    liquid_component_shells::Vector{Int32}
    liquid_component_volumes::Vector{Float64}
    cavity_component_volumes::Vector{Float64}
end

@inline function bounds_contain(outer::MeshShell, inner::MeshShell)
    scale = max(1.0, maximum(abs, outer.lower), maximum(abs, outer.upper),
                maximum(abs, inner.lower), maximum(abs, inner.upper))
    tolerance = 64eps(Float64) * scale
    return all(outer.lower .<= inner.lower .+ tolerance) &&
           all(outer.upper .>= inner.upper .- tolerance)
end

function shell_winding_number(mesh, shell::MeshShell, point::SVector{3, Float64})
    solid_angle = 0.0
    @inbounds for face_index in shell.face_indices
        face = mesh.faces[face_index]
        a = SVector{3, Float64}(mesh.vertices[face[1]]) - point
        b = SVector{3, Float64}(mesh.vertices[face[2]]) - point
        c = SVector{3, Float64}(mesh.vertices[face[3]]) - point
        a_norm, b_norm, c_norm = norm(a), norm(b), norm(c)
        minimum((a_norm, b_norm, c_norm)) > eps(Float64) ||
            error("surface shells touch or intersect at a containment probe")
        numerator = dot(a, cross(b, c))
        denominator = a_norm * b_norm * c_norm +
                      dot(a, b) * c_norm + dot(b, c) * a_norm +
                      dot(c, a) * b_norm
        solid_angle += 2atan(numerator, denominator)
    end
    return solid_angle / (4pi)
end

# Contiguous face-index range of a deterministic reduction bucket (see
# `PARALLEL_REDUCTION_BUCKETS`)
@inline function reduction_bucket_range(bucket, n_faces, nbuckets)
    chunk = cld(n_faces, nbuckets)
    return ((bucket - 1) * chunk + 1):min(bucket * chunk, n_faces)
end

# Number of deterministic reduction buckets for `n_faces` faces. Serial execution — or
# inputs too small for threading to pay off — uses one bucket, which reproduces the
# exact serial summation order.
function reduction_buckets(n_faces, backend)
    if backend isa SerialBackend || n_faces < PARALLEL_REDUCTION_MIN_FACES
        return 1
    end
    return PARALLEL_REDUCTION_BUCKETS
end

function mesh_liquid_analysis(mesh; zero_volume_face_indices=nothing,
                              backend=SerialBackend())
    parent = Int32.(eachindex(mesh.vertices))
    sizes = ones(Int32, length(parent))
    @inbounds for face in mesh.faces
        union_vertices!(parent, sizes, face[1], face[2])
        union_vertices!(parent, sizes, face[1], face[3])
    end

    # Per-component face lists in order of first appearance. Components keep their
    # discovery order, so everything downstream is unaffected by the accumulation order.
    # The unions above are complete, so `find_root!` only compresses paths from here on
    # and every `parent[face[1]]` below is already its root.
    component_by_root = zeros(Int32, length(mesh.vertices))
    roots = Int32[]
    face_indices = Vector{Int32}[]
    probes = SVector{3, Float64}[]
    references = SVector{3, Float64}[]
    @inbounds for (face_index, face) in enumerate(mesh.faces)
        root = find_root!(parent, face[1])
        # `find_root!` uses path halving, which need not set the starting vertex's
        # parent to the root. The parallel pass below directly indexes this entry.
        parent[face[1]] = root
        component = component_by_root[root]
        if component == 0
            a = SVector{3, Float64}(mesh.vertices[face[1]])
            b = SVector{3, Float64}(mesh.vertices[face[2]])
            c = SVector{3, Float64}(mesh.vertices[face[3]])
            push!(roots, root)
            push!(face_indices, Int32[])
            push!(probes, (a + b + c) / 3)
            push!(references, a)
            component = Int32(length(roots))
            component_by_root[root] = component
        end
        push!(face_indices[component], Int32(face_index))
    end
    ncomponents = length(roots)

    # Signed volumes and bounds accumulate per (component, bucket) over contiguous face
    # ranges. Bounds are exact under any order; volume sums agree up to floating-point
    # reassociation. Relative error depends on cancellation and the number of terms;
    # it has no input-independent bound. One bucket reproduces the serial order.
    nbuckets = reduction_buckets(length(mesh.faces), backend)
    volume_partials = zeros(Float64, ncomponents, nbuckets)
    lower_partials = fill(SVector{3, Float64}(Inf, Inf, Inf), ncomponents, nbuckets)
    upper_partials = fill(SVector{3, Float64}(-Inf, -Inf, -Inf), ncomponents, nbuckets)
    @threaded backend for bucket in 1:nbuckets
        @inbounds for face_index in reduction_bucket_range(bucket, length(mesh.faces),
                                             nbuckets)
            face = mesh.faces[face_index]
            a = SVector{3, Float64}(mesh.vertices[face[1]])
            b = SVector{3, Float64}(mesh.vertices[face[2]])
            c = SVector{3, Float64}(mesh.vertices[face[3]])
            component = component_by_root[parent[face[1]]]
            reference = references[component]
            volume_partials[component,
                            bucket] += dot(a - reference,
                                           cross(b - reference,
                                                 c - reference)) / 6
            triangle_lower = min.(a, min.(b, c))
            triangle_upper = max.(a, max.(b, c))
            lower_partials[component,
                           bucket] = min.(lower_partials[component, bucket],
                                          triangle_lower)
            upper_partials[component,
                           bucket] = max.(upper_partials[component, bucket],
                                          triangle_upper)
        end
    end
    signed_volumes = Vector{Float64}(undef, ncomponents)
    lower_bounds = Vector{SVector{3, Float64}}(undef, ncomponents)
    upper_bounds = Vector{SVector{3, Float64}}(undef, ncomponents)
    for component in 1:ncomponents
        signed_volume = volume_partials[component, 1]
        lower_bound = lower_partials[component, 1]
        upper_bound = upper_partials[component, 1]
        for bucket in 2:nbuckets
            signed_volume += volume_partials[component, bucket]
            lower_bound = min.(lower_bound, lower_partials[component, bucket])
            upper_bound = max.(upper_bound, upper_partials[component, bucket])
        end
        signed_volumes[component] = signed_volume
        lower_bounds[component] = lower_bound
        upper_bounds[component] = upper_bound
    end

    shells = MeshShell[]
    for component in eachindex(roots)
        signed_volume = signed_volumes[component]
        volume = abs(signed_volume)
        if volume <= eps(Float64)
            isnothing(zero_volume_face_indices) &&
                error("surface shell has zero signed volume")
            push!(zero_volume_face_indices, face_indices[component])
            continue
        end
        push!(shells,
              MeshShell(roots[component], face_indices[component], signed_volume, volume,
                        lower_bounds[component], upper_bounds[component],
                        probes[component]))
    end
    sort!(shells; by=shell -> (-shell.volume, shell.root))
    shell_by_root = Dict(shell.root => Int32(index)
                         for (index, shell) in enumerate(shells))

    shell_parents = zeros(Int32, length(shells))
    ascending_volume = sortperm(shells; by=shell -> shell.volume)
    for child_index in eachindex(shells)
        child = shells[child_index]
        for candidate_index in ascending_volume
            candidate = shells[candidate_index]
            candidate.volume > child.volume || continue
            bounds_contain(candidate, child) || continue
            if abs(shell_winding_number(mesh, candidate, child.probe)) > 0.5
                shell_parents[child_index] = Int32(candidate_index)
                break
            end
        end
    end

    shell_depths = zeros(Int32, length(shells))
    for shell_index in eachindex(shells)
        ancestor = shell_parents[shell_index]
        while ancestor != 0
            shell_depths[shell_index] += 1
            shell_depths[shell_index] <= length(shells) ||
                error("surface-shell containment hierarchy contains a cycle")
            ancestor = shell_parents[ancestor]
        end
    end

    child_volumes = zeros(Float64, length(shells))
    for child_index in eachindex(shells)
        shell_parent = shell_parents[child_index]
        shell_parent == 0 && continue
        child_volumes[shell_parent] += shells[child_index].volume
    end

    liquid_components = Tuple{Int32, Float64}[]
    cavity_components = Float64[]
    for shell_index in eachindex(shells)
        component_volume = shells[shell_index].volume - child_volumes[shell_index]
        component_volume > 0 ||
            error("nested shell has nonpositive domain volume")
        if iseven(shell_depths[shell_index])
            push!(liquid_components, (Int32(shell_index), component_volume))
        else
            push!(cavity_components, component_volume)
        end
    end
    sort!(liquid_components; by=last, rev=true)
    sort!(cavity_components; rev=true)
    return MeshLiquidAnalysis(parent, shells, shell_by_root, shell_parents, shell_depths,
                              first.(liquid_components), last.(liquid_components),
                              cavity_components)
end

function remove_face_components(mesh, face_components)
    discarded_faces = falses(length(mesh.faces))
    for face_indices in face_components, face_index in face_indices
        discarded_faces[face_index] = true
    end

    kept_faces = Face[]
    sizehint!(kept_faces, length(mesh.faces) - count(discarded_faces))
    used_vertices = falses(length(mesh.vertices))
    @inbounds for (face_index, face) in enumerate(mesh.faces)
        discarded_faces[face_index] && continue
        push!(kept_faces, face)
        used_vertices[face[1]] = true
        used_vertices[face[2]] = true
        used_vertices[face[3]] = true
    end

    remapping = zeros(Int32, length(mesh.vertices))
    vertices = SVector{3, Float32}[]
    sizehint!(vertices, count(used_vertices))
    @inbounds for (old_index, vertex) in enumerate(mesh.vertices)
        used_vertices[old_index] || continue
        push!(vertices, vertex)
        remapping[old_index] = Int32(length(vertices))
    end
    @inbounds for face_index in eachindex(kept_faces)
        face = kept_faces[face_index]
        kept_faces[face_index] = Face(remapping[face[1]], remapping[face[2]],
                                      remapping[face[3]])
    end
    return SurfaceMesh(vertices, kept_faces),
           (removed_components=length(face_components),
            removed_vertices=length(mesh.vertices) - length(vertices),
            removed_faces=count(discarded_faces))
end

function prune_zero_volume_shells(mesh; backend=SerialBackend())
    zero_volume_face_indices = Vector{Vector{Int32}}()
    analysis = mesh_liquid_analysis(mesh;
                                    zero_volume_face_indices=zero_volume_face_indices,
                                    backend)
    if isempty(zero_volume_face_indices)
        return mesh, analysis,
               (removed_components=0, removed_vertices=0, removed_faces=0)
    end

    pruned, removal = remove_face_components(mesh, zero_volume_face_indices)
    isempty(pruned.faces) && return pruned, nothing, removal
    return pruned, mesh_liquid_analysis(pruned; backend), removal
end

function remap_and_compact_mesh(mesh, remapping)
    remapped_faces = Face[]
    sizehint!(remapped_faces, length(mesh.faces))
    collapsed_faces = 0
    @inbounds for face in mesh.faces
        remapped = Face(remapping[face[1]], remapping[face[2]], remapping[face[3]])
        if distinct_vertices(remapped)
            push!(remapped_faces, remapped)
        else
            collapsed_faces += 1
        end
    end

    used_vertices = falses(length(mesh.vertices))
    @inbounds for face in remapped_faces
        used_vertices[face[1]] = true
        used_vertices[face[2]] = true
        used_vertices[face[3]] = true
    end
    compact_mapping = zeros(Int32, length(mesh.vertices))
    vertices = SVector{3, Float32}[]
    sizehint!(vertices, count(used_vertices))
    @inbounds for (old_index, vertex) in enumerate(mesh.vertices)
        used_vertices[old_index] || continue
        push!(vertices, vertex)
        compact_mapping[old_index] = Int32(length(vertices))
    end
    @inbounds for face_index in eachindex(remapped_faces)
        face = remapped_faces[face_index]
        remapped_faces[face_index] = Face(compact_mapping[face[1]],
                                          compact_mapping[face[2]],
                                          compact_mapping[face[3]])
    end
    return SurfaceMesh(vertices, remapped_faces),
           (removed_vertices=length(mesh.vertices) - length(vertices),
            removed_faces=collapsed_faces)
end

function topology_preserving_edge_collapse(mesh, first::Int32, second::Int32)
    incident_faces = Int32[]
    first_incidence = 0
    second_incidence = 0
    @inbounds for (face_index, face) in enumerate(mesh.faces)
        contains_first = first in face
        contains_second = second in face
        first_incidence += contains_first
        second_incidence += contains_second
        (contains_first || contains_second) && push!(incident_faces, Int32(face_index))
    end
    source, target = first_incidence <= second_incidence ?
                     (first, second) : (second, first)
    remapping = Int32.(eachindex(mesh.vertices))
    remapping[source] = target
    merged_vertex_link_is_manifold(mesh, remapping, target, incident_faces) ||
        return nothing
    collapsed, removal = remap_and_compact_mesh(mesh, remapping)
    return collapsed,
           merge(removal,
                 (edge_length=norm(SVector{3, Float64}(mesh.vertices[first]) -
                                   SVector{3, Float64}(mesh.vertices[second])),))
end

# Function barrier: `mesh` is reassigned in the collapse loop, so scanning it there would
# not be type-stable
function first_degenerate_face(mesh, area_tolerance)
    @inbounds for (face_index, face) in enumerate(mesh.faces)
        a = SVector{3, Float64}(mesh.vertices[face[1]])
        b = SVector{3, Float64}(mesh.vertices[face[2]])
        c = SVector{3, Float64}(mesh.vertices[face[3]])
        norm(cross(b - a, c - a)) <= 2area_tolerance && return Int32(face_index)
    end
    return nothing
end

function collapse_degenerate_triangles(mesh; area_tolerance=1.0e-14,
                                       maximum_collapses=1024)
    collapsed_edges = 0
    removed_vertices = 0
    removed_faces = 0
    maximum_edge_length = 0.0
    while true
        degenerate_face = first_degenerate_face(mesh, area_tolerance)
        isnothing(degenerate_face) && break
        collapsed_edges < maximum_collapses ||
            error("degenerate-triangle cleanup exceeded $maximum_collapses edge collapses")

        face = mesh.faces[degenerate_face]
        edges = [(face[1], face[2]), (face[2], face[3]), (face[3], face[1])]
        sort!(edges;
              by=edge -> norm(SVector{3, Float64}(mesh.vertices[edge[1]]) -
                              SVector{3, Float64}(mesh.vertices[edge[2]])))
        result = nothing
        for edge in edges
            result = topology_preserving_edge_collapse(mesh, edge[1], edge[2])
            isnothing(result) || break
        end
        isnothing(result) &&
            error("degenerate surface triangle cannot be removed by a topology-preserving edge collapse")
        mesh, removal = result
        collapsed_edges += 1
        removed_vertices += removal.removed_vertices
        removed_faces += removal.removed_faces
        maximum_edge_length = max(maximum_edge_length, removal.edge_length)
    end
    return mesh,
           (collapsed_edges=collapsed_edges,
            removed_vertices=removed_vertices,
            removed_faces=removed_faces,
            maximum_edge_length=maximum_edge_length)
end

function mesh_component_volumes(mesh)
    return mesh_liquid_analysis(mesh).liquid_component_volumes
end

# Numbers of undirected edges used by exactly one face (boundary edges) and by more than
# two faces (nonmanifold edges). Edges are grouped by their smaller vertex in compressed
# rows; the rows are short (about three edges per vertex), so duplicates are counted
# directly within each row.
function count_boundary_and_nonmanifold_edges(mesh)
    n_vertices = length(mesh.vertices)
    row_starts = zeros(Int, n_vertices + 1)
    @inbounds for face in mesh.faces
        row_starts[min(face[1], face[2])] += 1
        row_starts[min(face[2], face[3])] += 1
        row_starts[min(face[3], face[1])] += 1
    end
    # Exclusive prefix sum: row `vertex` is `row_starts[vertex]:(row_starts[vertex + 1] - 1)`
    position = 1
    @inbounds for vertex in 1:(n_vertices + 1)
        row_length = row_starts[vertex]
        row_starts[vertex] = position
        position += row_length
    end

    # Fill the rows with the larger vertex of every edge
    next_position = row_starts[1:n_vertices]
    other_vertices = Vector{eltype(eltype(mesh.faces))}(undef, 3 * length(mesh.faces))
    @inbounds for face in mesh.faces
        for (vertex_a, vertex_b) in ((face[1], face[2]), (face[2], face[3]),
             (face[3], face[1]))
            lower, upper = minmax(vertex_a, vertex_b)
            other_vertices[next_position[lower]] = upper
            next_position[lower] += 1
        end
    end

    boundary_edges = 0
    nonmanifold_edges = 0
    @inbounds for vertex in 1:n_vertices
        row_start, row_end = row_starts[vertex], row_starts[vertex + 1] - 1
        for position in row_start:row_end
            other = other_vertices[position]
            # Count every edge once, at its first occurrence in the row
            counted = false
            for previous in row_start:(position - 1)
                counted |= other_vertices[previous] == other
            end
            counted && continue
            uses = 0
            for next in position:row_end
                uses += other_vertices[next] == other
            end
            boundary_edges += uses == 1
            nonmanifold_edges += uses > 2
        end
    end
    return boundary_edges, nonmanifold_edges
end

# `analysis` optionally supplies the `mesh_liquid_analysis` of `mesh` (e.g. computed
# during the isovalue correction), which must then describe this exact mesh object
function mesh_geometry_stats(mesh; analysis=nothing, backend=SerialBackend())
    isnothing(analysis) && (analysis = mesh_liquid_analysis(mesh; backend))
    component_volumes = analysis.liquid_component_volumes
    # Deterministic parallel area sum over contiguous face ranges (see
    # `reduction_buckets`): one bucket reproduces the exact serial order
    nfaces = length(mesh.faces)
    nbuckets = reduction_buckets(nfaces, backend)
    area_partials = zeros(Float64, nbuckets)
    degenerate_partials = zeros(Int, nbuckets)
    @threaded backend for bucket in 1:nbuckets
        area_sum = 0.0
        degenerate_count = 0
        @inbounds for face_index in reduction_bucket_range(bucket, nfaces, nbuckets)
            face = mesh.faces[face_index]
            a = SVector{3, Float64}(mesh.vertices[face[1]])
            b = SVector{3, Float64}(mesh.vertices[face[2]])
            c = SVector{3, Float64}(mesh.vertices[face[3]])
            doubled_area = norm(cross(b - a, c - a))
            area_sum += doubled_area / 2
            degenerate_count += doubled_area <= 2.0e-14
        end
        area_partials[bucket] = area_sum
        degenerate_partials[bucket] = degenerate_count
    end
    area = area_partials[1]
    degenerate_triangles = degenerate_partials[1]
    for bucket in 2:nbuckets
        area += area_partials[bucket]
        degenerate_triangles += degenerate_partials[bucket]
    end
    boundary_edges, nonmanifold_edges = count_boundary_and_nonmanifold_edges(mesh)

    lower = SVector{3, Float64}(Inf, Inf, Inf)
    upper = SVector{3, Float64}(-Inf, -Inf, -Inf)
    for vertex in mesh.vertices
        point = SVector{3, Float64}(vertex)
        lower = min.(lower, point)
        upper = max.(upper, point)
    end
    return (volume=sum(component_volumes),
            surface_area=area,
            region_volumes=component_volumes,
            n_connected_regions=length(component_volumes),
            detached_region_volume=sum(@view component_volumes[2:end]),
            n_surface_components=length(analysis.shells),
            shell_volumes=[shell.volume for shell in analysis.shells],
            shell_signed_volumes=[iseven(depth) ? shell.volume : -shell.volume
                                  for (shell, depth) in zip(analysis.shells,
                                          analysis.shell_depths)],
            shell_nesting_depths=analysis.shell_depths,
            n_cavity_regions=length(analysis.cavity_component_volumes),
            cavity_volume=sum(analysis.cavity_component_volumes),
            n_boundary_edges=boundary_edges,
            n_nonmanifold_edges=nonmanifold_edges,
            n_degenerate_triangles=degenerate_triangles,
            lower=lower,
            upper=upper)
end

function vertices_inside_boundaries(mesh, boundaries; tolerance=1.0e-8,
                                    backend=PolyesterBackend())
    inside = zeros(UInt8, length(mesh.vertices))
    @threaded backend for index in eachindex(mesh.vertices)
        point = SVector{3, Float64}(mesh.vertices[index])
        for boundary in boundaries
            if point_in_bounds(point, boundary.lower, boundary.upper) &&
               signed_distance(boundary.bvh, point) < -tolerance
                inside[index] = 1
                break
            end
        end
    end
    return sum(inside)
end

function mesh_signed_volume(mesh)
    isempty(mesh.vertices) && return 0.0
    reference = SVector{3, Float64}(first(mesh.vertices))
    volume = 0.0
    @inbounds for face in mesh.faces
        a = SVector{3, Float64}(mesh.vertices[face[1]]) - reference
        b = SVector{3, Float64}(mesh.vertices[face[2]]) - reference
        c = SVector{3, Float64}(mesh.vertices[face[3]]) - reference
        volume += dot(a, cross(b, c)) / 6
    end
    return volume
end

function mesh_volume_centroid(mesh)
    analysis = mesh_liquid_analysis(mesh)
    liquid_volume = 0.0
    liquid_moment = SVector{3, Float64}(0, 0, 0)
    for (shell_index, shell) in enumerate(analysis.shells)
        reference = SVector{3, Float64}(mesh.vertices[mesh.faces[first(shell.face_indices)][1]])
        signed_moment = SVector{3, Float64}(0, 0, 0)
        @inbounds for face_index in shell.face_indices
            face = mesh.faces[face_index]
            a = SVector{3, Float64}(mesh.vertices[face[1]])
            b = SVector{3, Float64}(mesh.vertices[face[2]])
            c = SVector{3, Float64}(mesh.vertices[face[3]])
            signed_tetrahedron_volume = dot(a - reference,
                                            cross(b - reference, c - reference)) / 6
            signed_moment += signed_tetrahedron_volume *
                             (reference + a + b + c) / 4
        end
        shell_centroid = signed_moment / shell.signed_volume
        oriented_volume = iseven(analysis.shell_depths[shell_index]) ?
                          shell.volume : -shell.volume
        liquid_volume += oriented_volume
        liquid_moment += oriented_volume * shell_centroid
    end
    liquid_volume > eps(Float64) ||
        error("mesh liquid domain has nonpositive volume")
    return liquid_volume, liquid_moment / liquid_volume
end

# Per-vertex winding-reversal flags for a root-keyed `reverse_winding` table. Resolving
# the root once per vertex instead of once per face gives identical flags: `find_root!`
# returns the same root regardless of path compression.
function vertex_reverse_flags(parent, reverse_winding, n_vertices)
    flags = Vector{Bool}(undef, n_vertices)
    @inbounds for vertex in 1:n_vertices
        # Unreferenced vertices have no shell and never select a face winding. Their
        # normals use the fallback in `surface_vertex_normals`.
        flags[vertex] = get(reverse_winding, find_root!(parent, Int32(vertex)), false)
    end
    return flags
end

# `analysis` must describe the current contents of `mesh` when supplied. Despite the
# historical name, output coordinates are not reflected: outer shells must have positive
# signed volume, and odd-depth cavity shells negative signed volume.
function reflected_component_winding(mesh; analysis=nothing)
    isnothing(analysis) && (analysis = mesh_liquid_analysis(mesh))
    reverse_winding = Dict{Int32, Bool}()
    for (shell_index, shell) in enumerate(analysis.shells)
        desired_positive = iseven(analysis.shell_depths[shell_index])
        reverse_winding[shell.root] = (shell.signed_volume > 0) != desired_positive
    end
    return analysis.vertex_parent, reverse_winding
end

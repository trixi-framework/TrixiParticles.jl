# Closed planar contours, with outer loops counterclockwise and holes clockwise.
const ContourSegment = SVector{2, Int32}

struct ContourAnalysis
    loops::Vector{Vector{Int32}}
    depths::Vector{Int}
    signed_areas::Vector{Float64}
    liquid_component_volumes::Vector{Float64}
    oriented_faces::Vector{ContourSegment}
end

@inline contour_cross(a, b) = a[1] * b[2] - a[2] * b[1]

function contour_signed_area(vertices, loop)
    reference = SVector{2, Float64}(vertices[first(loop)])
    area = 0.0
    for index in eachindex(loop)
        a = SVector{2, Float64}(vertices[loop[index]]) - reference
        b = SVector{2, Float64}(vertices[loop[mod1(index + 1, length(loop))]]) - reference
        area += contour_cross(a, b) / 2
    end
    return area
end

function point_in_contour(point, vertices, loop)
    point = SVector{2, Float64}(point)
    inside = false
    for index in eachindex(loop)
        a = SVector{2, Float64}(vertices[loop[index]])
        b = SVector{2, Float64}(vertices[loop[mod1(index + 1, length(loop))]])
        if (a[2] > point[2]) != (b[2] > point[2])
            x_crossing = a[1] + (point[2] - a[2]) * (b[1] - a[1]) / (b[2] - a[2])
            inside = xor(inside, point[1] < x_crossing)
        end
    end
    return inside
end

@inline function contour_segments_intersect(a, b, c, d)
    a, b, c,
    d = SVector{2, Float64}(a), SVector{2, Float64}(b),
        SVector{2, Float64}(c), SVector{2, Float64}(d)
    any(max.(min.(a, b), min.(c, d)) .> min.(max.(a, b), max.(c, d))) && return false
    ab_c, ab_d = contour_cross(b - a, c - a), contour_cross(b - a, d - a)
    cd_a, cd_b = contour_cross(d - c, a - c), contour_cross(d - c, b - c)
    return ((ab_c <= 0 <= ab_d) || (ab_d <= 0 <= ab_c)) &&
           ((cd_a <= 0 <= cd_b) || (cd_b <= 0 <= cd_a))
end

function contour_analysis(mesh::SurfaceMesh{T, I, 2}; check_intersections=true) where {T, I}
    isempty(mesh.faces) && throw(ArgumentError("a contour needs line segments"))
    all(vertex -> all(isfinite, vertex), mesh.vertices) ||
        throw(ArgumentError("contour coordinates must be finite"))
    incident = [Int[] for _ in mesh.vertices]
    for (index, edge) in enumerate(mesh.faces)
        all(vertex -> 1 <= vertex <= length(mesh.vertices), edge) ||
            throw(ArgumentError("contour segment references a nonexistent vertex"))
        edge[1] != edge[2] && mesh.vertices[edge[1]] != mesh.vertices[edge[2]] ||
            throw(ArgumentError("contour contains a zero-length segment"))
        push!(incident[edge[1]], index)
        push!(incident[edge[2]], index)
    end
    all(edges -> isempty(edges) || length(edges) == 2, incident) ||
        throw(ArgumentError("contour must have degree two at every referenced vertex"))
    if check_intersections
        for index in eachindex(mesh.faces), other in (index + 1):length(mesh.faces)
            edge, next_edge = mesh.faces[index], mesh.faces[other]
            if any(vertex -> vertex in next_edge, edge)
                # Adjacent edges may share one endpoint, but not an entire segment.
                count(vertex -> vertex in next_edge, edge) == 1 ||
                    throw(ArgumentError("contour contains duplicate segments"))
                continue
            end
            contour_segments_intersect(mesh.vertices[edge[1]], mesh.vertices[edge[2]],
                                       mesh.vertices[next_edge[1]],
                                       mesh.vertices[next_edge[2]]) &&
                throw(ArgumentError("contour loops must not intersect or touch"))
        end
    end
    visited = falses(length(mesh.faces))
    loops = Vector{Int32}[]
    loop_edges = Vector{Int}[]
    for first_edge in eachindex(mesh.faces)
        visited[first_edge] && continue
        start = mesh.faces[first_edge][1]
        vertex, edge_index = start, first_edge
        loop = Int32[]
        edge_indices = Int[]
        while true
            visited[edge_index] && throw(ArgumentError("contour is not a simple cycle"))
            push!(loop, Int32(vertex))
            push!(edge_indices, edge_index)
            visited[edge_index] = true
            edge = mesh.faces[edge_index]
            vertex = edge[1] == vertex ? edge[2] : edge[1]
            vertex == start && break
            adjacent = incident[vertex]
            edge_index = adjacent[1] == edge_index ? adjacent[2] : adjacent[1]
        end
        length(loop) >= 3 || throw(ArgumentError("a closed contour needs three vertices"))
        push!(loops, loop)
        push!(loop_edges, edge_indices)
    end
    signed_areas = [contour_signed_area(mesh.vertices, loop) for loop in loops]
    all(area -> isfinite(area) && !iszero(area), signed_areas) ||
        throw(ArgumentError("contour loops must enclose nonzero finite area"))
    # Nonintersecting loops are either disjoint or nested. A point on a child's boundary
    # is strictly inside each enclosing loop, so parity gives liquid versus hole.
    depths = [count(other -> other != index &&
                             point_in_contour(mesh.vertices[first(loop)], mesh.vertices,
                                              loops[other]),
                    eachindex(loops)) for (index, loop) in enumerate(loops)]
    region_areas = Float64[]
    for index in eachindex(loops)
        isodd(depths[index]) && continue
        area = abs(signed_areas[index])
        for child in eachindex(loops)
            depths[child] == depths[index] + 1 || continue
            if point_in_contour(mesh.vertices[first(loops[child])], mesh.vertices,
                                loops[index])
                area -= abs(signed_areas[child])
            end
        end
        area > 0 || throw(ArgumentError("nested contours leave no positive liquid area"))
        push!(region_areas, area)
    end
    sort!(region_areas; rev=true)
    # Preserve input edge ordering. Reanalyzing an already-oriented contour can traverse
    # its hole loops in the opposite order; that must not reorder cells in cached vs
    # uncached output files.
    oriented_faces = Vector{ContourSegment}(undef, length(mesh.faces))
    for (index, loop) in enumerate(loops)
        reverse_loop = (signed_areas[index] > 0) != iseven(depths[index])
        for vertex in eachindex(loop)
            a, b = loop[vertex], loop[mod1(vertex + 1, length(loop))]
            oriented_faces[loop_edges[index][vertex]] = reverse_loop ?
                                                        ContourSegment(b, a) :
                                                        ContourSegment(a, b)
        end
    end
    return ContourAnalysis(loops, depths, signed_areas, region_areas, oriented_faces)
end

function mesh_geometry_stats(mesh::SurfaceMesh{T, I, 2}; analysis=nothing,
                             backend=SerialBackend()) where {T, I}
    analysis = isnothing(analysis) ? contour_analysis(mesh) : analysis
    areas = analysis.liquid_component_volumes
    perimeter = sum(norm(SVector{2, Float64}(mesh.vertices[edge[2]]) -
                         SVector{2, Float64}(mesh.vertices[edge[1]]))
                    for edge in mesh.faces)
    lower = reduce((a, b) -> min.(a, b), mesh.vertices)
    upper = reduce((a, b) -> max.(a, b), mesh.vertices)
    return (; volume=sum(areas), surface_area=perimeter,
            region_volumes=areas, n_connected_regions=length(areas),
            detached_region_volume=sum(areas[2:end]; init=0.0),
            n_surface_components=length(analysis.loops),
            shell_volumes=abs.(analysis.signed_areas),
            shell_signed_volumes=[iseven(depth) ? abs(area) : -abs(area)
                                  for (depth, area) in
                                      zip(analysis.depths, analysis.signed_areas)],
            shell_nesting_depths=analysis.depths,
            n_cavity_regions=count(isodd, analysis.depths),
            cavity_volume=sum(abs(area)
                              for (depth, area) in
                                  zip(analysis.depths, analysis.signed_areas)
                              if isodd(depth); init=0.0),
            n_boundary_edges=0, n_nonmanifold_edges=0, n_degenerate_triangles=0, lower,
            upper)
end

function surface_vertex_normals(mesh::SurfaceMesh{T, I, 2}; analysis=nothing) where {T, I}
    analysis = isnothing(analysis) ? contour_analysis(mesh) : analysis
    normals = fill(SVector(0.0, 0.0), length(mesh.vertices))
    for edge in analysis.oriented_faces
        tangent = SVector{2, Float64}(mesh.vertices[edge[2]]) -
                  SVector{2, Float64}(mesh.vertices[edge[1]])
        normal = SVector(Float64(tangent[2]), -Float64(tangent[1]))
        normals[edge[1]] += normal
        normals[edge[2]] += normal
    end
    for index in eachindex(normals)
        magnitude = norm(normals[index])
        normals[index] = magnitude > 0 ? normals[index] / magnitude : SVector(0.0, 1.0)
    end
    return normals, analysis
end

struct SegmentBoundary
    vertices::Vector{SVector{2, Float64}}
    analysis::ContourAnalysis
end

function signed_distance(boundary::SegmentBoundary, point)
    best = Inf
    for edge in boundary.analysis.oriented_faces
        a, b = boundary.vertices[edge[1]], boundary.vertices[edge[2]]
        direction = b - a
        fraction = clamp(dot(point - a, direction) / dot(direction, direction), 0.0, 1.0)
        best = min(best, norm(point - (a + fraction * direction)))
    end
    iszero(best) && return best
    inside = isodd(count(loop -> point_in_contour(point, boundary.vertices, loop),
                         boundary.analysis.loops))
    return inside ? -best : best
end

function BoundaryMesh(points, topology::BoundaryTopology{2})
    ndims(points) == 2 && size(points, 1) == 2 ||
        throw(ArgumentError("2D boundaries need a 2×n coordinate matrix"))
    coordinates = Matrix{Float64}(points)
    vertices = SVector{2, Float64}.(eachcol(coordinates))
    analysis = contour_analysis(SurfaceMesh(vertices, topology.faces))
    lower = reduce((a, b) -> min.(a, b), vertices)
    upper = reduce((a, b) -> max.(a, b), vertices)
    return BoundaryMesh(coordinates, analysis.oriented_faces,
                        SegmentBoundary(vertices, analysis), lower, upper)
end

function BoundaryMesh(polygon::Polygon{2})
    points = reduce(hcat, polygon.vertices)
    # Polygon may repeat its first vertex at the end. Refer to the original first
    # vertex instead so the contour graph is closed topologically as well as spatially.
    last_id = length(polygon.vertices)
    repeated = isapprox(polygon.vertices[1], polygon.vertices[end])
    remap(i) = repeated && i == last_id ? 1 : i
    faces = [ContourSegment(remap(a), remap(b)) for (a, b) in polygon.edge_vertices_ids]
    return BoundaryMesh(points, BoundaryTopology(faces))
end

function BoundaryMesh(mesh::SurfaceMesh{T, I, 2}) where {T, I}
    return BoundaryMesh(reduce(hcat, mesh.vertices),
                        BoundaryTopology(ContourSegment.(mesh.faces)))
end

function boundary_surface_mesh(boundary::BoundaryMesh{2})
    used = sort!(unique(vcat([collect(edge) for edge in boundary.faces]...)))
    index = Dict(old => Int32(new) for (new, old) in enumerate(used))
    vertices = [SVector{2, Float64}(boundary.points[:, vertex]) for vertex in used]
    faces = [ContourSegment(index[edge[1]], index[edge[2]]) for edge in boundary.faces]
    return SurfaceMesh(vertices, faces)
end

function lattice_contour_topology(reference)
    lookup = lattice_coordinate_lookup(reference, Val(2))
    nx, ny = size(lookup)
    loop = vcat(lookup[:, 1], lookup[nx, 2:end], reverse(lookup[1:(nx - 1), ny]),
                reverse(lookup[1, 2:(ny - 1)]))
    return BoundaryTopology([ContourSegment(loop[i], loop[mod1(i + 1, length(loop))])
                             for i in eachindex(loop)])
end

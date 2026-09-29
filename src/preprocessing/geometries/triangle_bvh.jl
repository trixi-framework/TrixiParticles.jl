# Nearest-triangle signed-distance acceleration over closed triangle meshes.
#
# A bounding volume hierarchy (BVH) over the mesh triangles with angle-weighted
# pseudonormals (vertex/edge/face) following Christer Ericson's Real-Time Collision
# Detection. Used for implicit boundary constraints and inside/outside classification.
# The arithmetic is intentionally self-contained so that validated results stay
# bitwise reproducible; see `signed_distance`.
@inline point3(points,
               index) = SVector{3, Float64}(points[1, index], points[2, index],
                                            points[3, index])

# Key of the undirected edge between two vertices
@inline function edge_key(first::Int32, second::Int32)
    lower, upper = minmax(first, second)
    return (UInt64(reinterpret(UInt32, lower)) << 32) |
           UInt64(reinterpret(UInt32, upper))
end

struct BvhTriangle
    a::SVector{3, Float64}
    b::SVector{3, Float64}
    c::SVector{3, Float64}
    normal::SVector{3, Float64}
    vertex_normals::NTuple{3, SVector{3, Float64}}
    edge_normals::NTuple{3, SVector{3, Float64}}
    centroid::SVector{3, Float64}
    lower::SVector{3, Float64}
    upper::SVector{3, Float64}
end

struct BvhNode
    lower::SVector{3, Float64}
    upper::SVector{3, Float64}
    left::Int32
    right::Int32
    first::Int32
    count::Int32
end

"""
    TriangleBvh

Bounding volume hierarchy over closed triangle meshes for signed-distance queries
using closest-point distances to triangles (up to floating-point roundoff). Build with
[`build_triangle_bvh`](@ref); query with `signed_distance`.
"""
struct TriangleBvh
    triangles::Vector{BvhTriangle}
    order::Vector{Int32}
    nodes::Vector{BvhNode}
end

"""
    build_triangle_bvh(points, faces; leaf_size=8)

Build a [`TriangleBvh`](@ref) over triangles with vertices in the columns of `points`
(3×n matrix) and 1-based triangular `faces`. Vertex order is preserved exactly, so
results stay bitwise reproducible for a given input ordering.
"""
function build_triangle_bvh(points, faces; leaf_size=8)
    size(points, 1) == 3 && all(isfinite, points) ||
        throw(ArgumentError("triangle mesh points must be a finite 3×n matrix"))
    !isempty(faces) && leaf_size >= 1 ||
        throw(ArgumentError("a triangle BVH needs faces and a positive leaf size"))
    face_normals = Vector{SVector{3, Float64}}(undef, length(faces))
    vertex_normal_sums = fill(SVector{3, Float64}(0, 0, 0), size(points, 2))
    used_vertices = falses(size(points, 2))
    edge_normal_sums = Dict{UInt64, SVector{3, Float64}}()
    edge_uses = Dict{UInt64, Tuple{Int, Int}}()
    for (index, face) in enumerate(faces)
        all(vertex -> 1 <= vertex <= size(points, 2), face) ||
            throw(ArgumentError("triangle mesh face references a nonexistent vertex"))
        a, b, c = (point3(points, face[axis]) for axis in 1:3)
        normal = cross(b - a, c - a)
        magnitude = norm(normal)
        magnitude > eps(Float64) ||
            throw(ArgumentError("triangle mesh contains a degenerate triangle"))
        face_normal = normal / magnitude
        face_normals[index] = face_normal
        for (vertex, first, second) in ((face[1], b - a, c - a),
             (face[2], c - b, a - b),
             (face[3], a - c, b - c))
            used_vertices[vertex] = true
            cosine = clamp(dot(first, second) / (norm(first) * norm(second)), -1.0, 1.0)
            vertex_normal_sums[vertex] += acos(cosine) * face_normal
        end
        for (first, second) in ((face[1], face[2]), (face[2], face[3]),
                                (face[3], face[1]))
            key = edge_key(first, second)
            edge_normal_sums[key] = get(edge_normal_sums, key,
                                        SVector{3, Float64}(0, 0, 0)) +
                                    face_normal
            count, orientation = get(edge_uses, key, (0, 0))
            edge_uses[key] = (count + 1, orientation + (first < second ? 1 : -1))
        end
    end
    all(==((2, 0)), values(edge_uses)) ||
        throw(ArgumentError("signed distances require a closed, consistently oriented triangle mesh"))
    vertex_normals = map(eachindex(vertex_normal_sums)) do index
        used_vertices[index] || return SVector{3, Float64}(0, 0, 0)
        normal = vertex_normal_sums[index]
        magnitude = norm(normal)
        magnitude > eps(Float64) ||
            throw(ArgumentError("triangle mesh vertex has an undefined pseudonormal"))
        normal / magnitude
    end
    edge_normals = Dict(key => begin
                            magnitude = norm(normal)
                            magnitude > eps(Float64) ||
                                throw(ArgumentError("triangle mesh edge has an undefined pseudonormal"))
                            normal / magnitude
                        end for (key, normal) in edge_normal_sums)

    triangles = Vector{BvhTriangle}(undef, length(faces))
    for (index, face) in enumerate(faces)
        a, b, c = (point3(points, face[axis]) for axis in 1:3)
        lower = min.(a, min.(b, c))
        upper = max.(a, max.(b, c))
        triangles[index] = BvhTriangle(a, b, c, face_normals[index],
                                       (vertex_normals[face[1]],
                                        vertex_normals[face[2]],
                                        vertex_normals[face[3]]),
                                       (edge_normals[edge_key(face[1], face[2])],
                                        edge_normals[edge_key(face[2], face[3])],
                                        edge_normals[edge_key(face[3], face[1])]),
                                       (a + b + c) / 3, lower, upper)
    end
    order = Int32.(eachindex(triangles))
    nodes = BvhNode[]
    sizehint!(nodes, 2 * length(triangles))

    function build_node!(first, last)
        lower = SVector{3, Float64}(Inf, Inf, Inf)
        upper = SVector{3, Float64}(-Inf, -Inf, -Inf)
        centroid_lower = lower
        centroid_upper = upper
        for order_index in first:last
            triangle = triangles[order[order_index]]
            lower = min.(lower, triangle.lower)
            upper = max.(upper, triangle.upper)
            centroid_lower = min.(centroid_lower, triangle.centroid)
            centroid_upper = max.(centroid_upper, triangle.centroid)
        end
        node_index = length(nodes) + 1
        push!(nodes, BvhNode(lower, upper, 0, 0, Int32(first), Int32(last - first + 1)))
        if last - first + 1 > leaf_size
            extent = centroid_upper - centroid_lower
            axis = argmax(extent)
            sort!(view(order, first:last); by=index -> triangles[index].centroid[axis])
            middle = (first + last) >>> 1
            left = build_node!(first, middle)
            right = build_node!(middle + 1, last)
            nodes[node_index] = BvhNode(lower, upper, Int32(left), Int32(right), 0, 0)
        end
        return node_index
    end

    build_node!(1, length(order))
    return TriangleBvh(triangles, order, nodes)
end

@inline function aabb_distance_squared(point, lower, upper)
    distance = 0.0
    @inbounds for axis in 1:3
        delta = point[axis] < lower[axis] ? lower[axis] - point[axis] :
                point[axis] > upper[axis] ? point[axis] - upper[axis] : 0.0
        distance = muladd(delta, delta, distance)
    end
    return distance
end

# Closest-point regions follow Christer Ericson's Real-Time Collision Detection.
@inline function closest_point_and_pseudonormal(point, triangle)
    return triangle_closest_point_and_normal(point, triangle.a, triangle.b, triangle.c,
                                             triangle.vertex_normals, triangle.edge_normals,
                                             triangle.normal, Val(true))
end

"""
    signed_distance(bvh, point)

Signed Euclidean distance from `point` to the closed mesh in `bvh` (negative inside),
up to floating-point roundoff. The mesh must be embedded (no self-intersections) and
oriented out of the solid, including inward-facing cavity shells. Angle-weighted
pseudonormals determine the sign near edges and vertices under these assumptions.
"""
function signed_distance(bvh::TriangleBvh, point::SVector{3, Float64})
    stack = MVector{64, Int32}(undef)
    stack_size = 1
    stack[1] = 1
    best_distance_squared = Inf
    best_side = 1.0
    while stack_size > 0
        node_index = stack[stack_size]
        stack_size -= 1
        node = bvh.nodes[node_index]
        aabb_distance_squared(point, node.lower, node.upper) <= best_distance_squared ||
            continue
        if node.count > 0
            first = Int(node.first)
            last = first + Int(node.count) - 1
            @inbounds for order_index in first:last
                triangle = bvh.triangles[bvh.order[order_index]]
                closest, pseudonormal = closest_point_and_pseudonormal(point, triangle)
                offset = point - closest
                distance_squared = dot(offset, offset)
                side = dot(offset, pseudonormal)
                if !isfinite(best_distance_squared) ||
                   distance_squared < best_distance_squared ||
                   (distance_squared == best_distance_squared &&
                    abs(side) > abs(best_side))
                    best_distance_squared = distance_squared
                    best_side = side
                end
            end
        else
            left = bvh.nodes[node.left]
            right = bvh.nodes[node.right]
            left_distance = aabb_distance_squared(point, left.lower, left.upper)
            right_distance = aabb_distance_squared(point, right.lower, right.upper)
            if left_distance < right_distance
                if right_distance <= best_distance_squared
                    stack_size += 1;
                    stack[stack_size] = node.right
                end
                if left_distance <= best_distance_squared
                    stack_size += 1;
                    stack[stack_size] = node.left
                end
            else
                if left_distance <= best_distance_squared
                    stack_size += 1;
                    stack[stack_size] = node.left
                end
                if right_distance <= best_distance_squared
                    stack_size += 1;
                    stack[stack_size] = node.right
                end
            end
        end
    end
    return copysign(sqrt(best_distance_squared), best_side)
end

"""
    build_triangle_bvh(geometry::TriangleMesh{3}; leaf_size=8)

Build a [`TriangleBvh`](@ref) over the mesh vertices in mesh order. Vertices are
converted to `Float64` exactly; faces keep the mesh's winding. For lattice particle
sets, use `BoundaryMesh` with `lattice_surface_topology` instead, which preserves
the particle ordering.
"""
function build_triangle_bvh(geometry::TriangleMesh{3}; leaf_size=8)
    points = Matrix{Float64}(undef, 3, length(geometry.vertices))
    for (index, vertex) in enumerate(geometry.vertices)
        points[:, index] = vertex
    end
    # The geometry layer must not depend on the reconstruction-specific `Face` alias.
    faces = [SVector{3, Int32}(ids...) for ids in geometry.face_vertices_ids]
    return build_triangle_bvh(points, faces; leaf_size=leaf_size)
end

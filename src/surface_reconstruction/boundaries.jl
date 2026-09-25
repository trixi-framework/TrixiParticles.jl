# Boundary meshes: lattice surface topology, closed-mesh containers, enclosure queries.
# Nearest-triangle signed distances come from `TriangleBvh` (see `preprocessing/geometries`).
"""
    BoundaryTopology

Triangle (3D) or line-segment (2D) connectivity of a closed boundary, independent of positions.
Built once by [`lattice_surface_topology`](@ref) from a reference configuration and
reused with the current coordinates of a moving boundary in [`BoundaryMesh`](@ref).
"""
struct BoundaryTopology{NDIMS}
    faces::Vector{SVector{NDIMS, Int32}}
end

function add_quad!(faces, lookup, a, b, c, d, reverse)
    vertices = reverse ? (a, d, c, b) : (a, b, c, d)
    indices = ntuple(i -> lookup[vertices[i]...], 4)
    push!(faces, Face(indices[1], indices[2], indices[3]))
    push!(faces, Face(indices[1], indices[3], indices[4]))
    return nothing
end

"""
    lattice_surface_topology(reference)

Closed surface of a complete 2D or 3D particle lattice (e.g. the initial
configuration of a `TotalLagrangianSPHSystem`): one quad per lattice cell on the surface,
split into two triangles with outward winding in 3D, or counterclockwise perimeter
segments in 2D. Combine with current coordinates via
[`BoundaryMesh`](@ref) to track a moving boundary.
"""
function lattice_surface_topology(reference)
    ndims(reference) == 2 && size(reference, 1) == 2 &&
        return lattice_contour_topology(reference)
    ndims(reference) == 2 && size(reference, 1) == 3 && all(isfinite, reference) ||
        throw(ArgumentError("reference coordinates must be a finite 3×n matrix"))
    grid_axes = ntuple(axis -> sort!(unique(round.(reference[axis, :]; digits=10))), 3)
    dimensions = length.(grid_axes)
    all(>=(2), dimensions) ||
        throw(ArgumentError("a closed lattice surface needs at least two points per axis"))
    prod(dimensions) == size(reference, 2) ||
        throw(ArgumentError("reference coordinates do not form a complete lattice"))
    axis_lookup = ntuple(axis -> Dict(value => Int32(index)
                                      for (index, value) in enumerate(grid_axes[axis])), 3)
    lookup = zeros(Int32, dimensions...)
    for point_index in axes(reference, 2)
        lattice_index = ntuple(axis -> axis_lookup[axis][round(reference[axis, point_index];
                                                               digits=10)], 3)
        iszero(lookup[lattice_index...]) ||
            throw(ArgumentError("reference lattice contains duplicate coordinates"))
        lookup[lattice_index...] = Int32(point_index)
    end

    nx, ny, nz = dimensions
    faces = Face[]
    sizehint!(faces, 4 * ((nx - 1) * (ny - 1) + (nx - 1) * (nz - 1) +
                          (ny - 1) * (nz - 1)))
    for x in (1, nx), y in 1:(ny - 1), z in 1:(nz - 1)
        add_quad!(faces, lookup, (x, y, z), (x, y + 1, z),
                  (x, y + 1, z + 1), (x, y, z + 1), x == 1)
    end
    for y in (1, ny), x in 1:(nx - 1), z in 1:(nz - 1)
        add_quad!(faces, lookup, (x, y, z), (x, y, z + 1),
                  (x + 1, y, z + 1), (x + 1, y, z), y == 1)
    end
    for z in (1, nz), x in 1:(nx - 1), y in 1:(ny - 1)
        add_quad!(faces, lookup, (x, y, z), (x + 1, y, z),
                  (x + 1, y + 1, z), (x, y + 1, z), z == 1)
    end
    return BoundaryTopology(faces)
end

struct BoundaryMesh{NDIMS, BVH}
    points::Matrix{Float64}
    faces::Vector{SVector{NDIMS, Int32}}
    bvh::BVH
    lower::SVector{NDIMS, Float64}
    upper::SVector{NDIMS, Float64}
end

Base.ndims(::BoundaryMesh{N}) where {N} = N

"""
    BoundaryMesh(points, topology)
    BoundaryMesh(geometry::TriangleMesh{3})
    BoundaryMesh(geometry::Polygon{2})
    BoundaryMesh(contour::SurfaceMesh{<:Any, <:Any, 2})

Closed surface of a fluid boundary — a wall or a structure — with a
signed-distance representation (triangle BVH in 3D, closed segments in 2D), used to clip
the reconstructed free surface and for inside/outside queries. Pass particle coordinates
(2×n or 3×n matrix) with a
[`BoundaryTopology`](@ref) — typically [`lattice_surface_topology`](@ref) of the initial
configuration combined with current coordinates — or any
[`TrixiParticles.TriangleMesh`](@ref), e.g. from [`load_geometry`](@ref). Vertices keep
their input ordering, so results stay bitwise reproducible for a given input. The surface
must be non-self-intersecting and
oriented out of the solid (into cavities for cavity shells). Open edges and inconsistent
edge orientations are rejected; geometric self-intersections are not detected.
"""
function BoundaryMesh(points, topology::BoundaryTopology{3})
    lower = SVector{3, Float64}(minimum(@view(points[1, :])), minimum(@view(points[2, :])),
                                minimum(@view(points[3, :])))
    upper = SVector{3, Float64}(maximum(@view(points[1, :])), maximum(@view(points[2, :])),
                                maximum(@view(points[3, :])))
    coordinates = points isa Matrix{Float64} ? points : Matrix{Float64}(points)
    return BoundaryMesh(coordinates, topology.faces,
                        build_triangle_bvh(points, topology.faces), lower, upper)
end

function BoundaryMesh(geometry::TriangleMesh{3})
    points = Matrix{Float64}(undef, 3, length(geometry.vertices))
    for (index, vertex) in enumerate(geometry.vertices)
        points[:, index] = vertex
    end
    return BoundaryMesh(points,
                        BoundaryTopology([Face(ids...)
                                          for ids in geometry.face_vertices_ids]))
end

@inline function point_in_bounds(point, lower, upper)
    return lower[1] <= point[1] <= upper[1] &&
           lower[2] <= point[2] <= upper[2] &&
           lower[3] <= point[3] <= upper[3]
end

"""
    enclosed_particles(points, boundaries; backend=PolyesterBackend())

Indices of `points` (as a `UInt8` mask) strictly inside any closed boundary mesh, e.g.
to exclude fluid particles covered by a wall or structure before reconstruction. Distances come
from each boundary's exact BVH; the axis-aligned bounds reject outside points cheaply.
"""
function enclosed_particles(points, boundaries; backend=PolyesterBackend())
    ndims(points) == 2 && size(points, 1) in (2, 3) ||
        throw(ArgumentError("particle coordinates must be a 2×n or 3×n matrix"))
    all(boundary -> boundary isa BoundaryMesh && ndims(boundary) == size(points, 1),
        boundaries) ||
        throw(ArgumentError("boundary and particle dimensions must match"))
    size(points, 1) == 2 && return enclosed_particles_2d(points, boundaries; backend)
    enclosed = zeros(UInt8, size(points, 2))
    @threaded backend for index in axes(points, 2)
        point = point3(points, index)
        for boundary in boundaries
            if point_in_bounds(point, boundary.lower, boundary.upper) &&
               signed_distance(boundary.bvh, point) < -1.0e-10
                enclosed[index] = 1
                break
            end
        end
    end
    return enclosed
end

# Surface mesh of a boundary for output. Only vertices referenced by faces are kept, so
# interior lattice particles do not appear as stray points.
function boundary_surface_mesh(boundary)
    new_index = zeros(Int32, size(boundary.points, 2))
    vertices = SVector{3, Float32}[]
    for face in boundary.faces, vertex in face
        if new_index[vertex] == 0
            push!(vertices,
                  SVector{3, Float32}(boundary.points[1, vertex],
                                      boundary.points[2, vertex],
                                      boundary.points[3, vertex]))
            new_index[vertex] = length(vertices)
        end
    end
    faces = [Face(new_index[face[1]], new_index[face[2]], new_index[face[3]])
             for face in boundary.faces]
    return SurfaceMesh(vertices, faces)
end

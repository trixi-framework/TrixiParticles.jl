include("polygon.jl")
include("triangle_mesh.jl")
include("io.jl")

@inline eachface(mesh) = Base.OneTo(nfaces(mesh))

"""
    is_closed_geometry(geometry)

Return `true` if a polygon or triangle mesh forms a closed region or surface.
"""
function is_closed_geometry(polygon::Polygon)
    vertices_close = isapprox(first(polygon.vertices), last(polygon.vertices))
    vertices_close || return false

    expected_edges = count(1:(length(polygon.vertices) - 1)) do i
        !isapprox(polygon.vertices[i], polygon.vertices[i + 1])
    end

    return nfaces(polygon) == expected_edges
end

function is_closed_geometry(mesh::TriangleMesh)
    return all(==(2), edge_face_counts(mesh))
end

function require_closed_geometry(geometry, operation)
    is_closed_geometry(geometry) && return nothing

    msg = "`$operation` requires a closed geometry. " *
          closure_error_detail(geometry)

    throw(ArgumentError(msg))
end

function closure_error_detail(polygon::Polygon)
    if !isapprox(first(polygon.vertices), last(polygon.vertices))
        return "The first and last polygon vertices are different. " *
               "If the vertices already trace a complete 2D boundary, construct or load " *
               "the geometry with `close_curve=true`; otherwise provide a closed boundary."
    end

    return "The polygon edge list does not form a complete closed curve."
end

function closure_error_detail(mesh::TriangleMesh)
    invalid_edges = count(!=(2), edge_face_counts(mesh))

    return "Found $invalid_edges mesh edges with an incident-face count different from 2."
end

function edge_face_counts(mesh::TriangleMesh)
    edge_face_counts = zeros(Int, length(mesh.edge_vertices_ids))

    for face_edges in mesh.face_edges_ids
        edge_face_counts[face_edges[1]] += 1
        edge_face_counts[face_edges[2]] += 1
        edge_face_counts[face_edges[3]] += 1
    end

    return edge_face_counts
end

function Base.setdiff(initial_condition::InitialCondition,
                      geometries::Union{Polygon, TriangleMesh}...)
    geometry = first(geometries)

    if ndims(geometry) != ndims(initial_condition)
        throw(ArgumentError("all passed geometries must have the same dimensionality as the initial condition"))
    end
    require_closed_geometry(geometry, "setdiff")

    coords = reinterpret(reshape,
                         SVector{ndims(geometry), eltype(initial_condition.coordinates)},
                         initial_condition.coordinates)

    delete_indices, _ = WindingNumberJacobson(; geometry)(geometry, coords)

    coordinates = initial_condition.coordinates[:, .!delete_indices]
    velocity = initial_condition.velocity[:, .!delete_indices]
    mass = initial_condition.mass[.!delete_indices]
    density = initial_condition.density[.!delete_indices]
    pressure = initial_condition.pressure[.!delete_indices]

    result = InitialCondition{ndims(initial_condition)}(coordinates, velocity, mass,
                                                        density, pressure,
                                                        initial_condition.particle_spacing)

    return setdiff(result, Base.tail(geometries)...)
end

function Base.intersect(initial_condition::InitialCondition,
                        geometries::Union{Polygon, TriangleMesh}...)
    geometry = first(geometries)

    if ndims(geometry) != ndims(initial_condition)
        throw(ArgumentError("all passed geometries must have the same dimensionality as the initial condition"))
    end
    require_closed_geometry(geometry, "intersect")

    coords = reinterpret(reshape,
                         SVector{ndims(geometry), eltype(initial_condition.coordinates)},
                         initial_condition.coordinates)

    keep_indices, _ = WindingNumberJacobson(; geometry)(geometry, coords)

    coordinates = initial_condition.coordinates[:, keep_indices]
    velocity = initial_condition.velocity[:, keep_indices]
    mass = initial_condition.mass[keep_indices]
    density = initial_condition.density[keep_indices]
    pressure = initial_condition.pressure[keep_indices]

    result = InitialCondition{ndims(initial_condition)}(coordinates, velocity, mass,
                                                        density, pressure,
                                                        initial_condition.particle_spacing)

    return intersect(result, Base.tail(geometries)...)
end

"""
    planar_geometry_to_face(planar_geometry::TriangleMesh)

Extracts a simplified rectangular face and its normal vector from an arbitrary planar geometry
(`TriangleMesh` loaded via [`load_geometry`](@ref))
for use as a boundary zone interface in [`BoundaryZone`](@ref).
This function computes the corner points of an oriented bounding box
that best represents the essential orientation and extent of the input geometry.
The geometry must be planar (all vertices should lie in the same plane),
but can have complex or non-rectangular boundaries.

!!! note "Face Normal Orientation"
    All face normals of the input geometry must point inside the fluid domain.
    The returned plane normal is computed by averaging all face normals, so consistent orientation is required.

# Arguments
- `planar_geometry`: A planar geometry (`TriangleMesh` loaded via [`load_geometry`](@ref)).

# Returns
- `face_vertices`: Tuple of three vertices defining the rectangular face (corner points of the oriented bounding box).
- `face_normal`: Normalized normal vector of the face.

# Example
```jldoctest; output=false, filter=r"face = .*"
file = pkgdir(TrixiParticles, "test", "preprocessing", "data")
planar_geometry = load_geometry(joinpath(file, "inflow_geometry.stl"))

face, face_normal = planar_geometry_to_face(planar_geometry)

# output
(face = ...)
```
"""
function planar_geometry_to_face(planar_geometry::TriangleMesh)
    face_normal = normalize(sum(planar_geometry.face_normals) / nfaces(planar_geometry))

    face_vertices = oriented_bounding_box(stack(planar_geometry.vertices))

    # Vectors spanning the face
    edge1 = face_vertices[:, 2] - face_vertices[:, 1]
    edge2 = face_vertices[:, 3] - face_vertices[:, 1]

    if abs(dot(edge1, face_normal)) > 1e-2 || abs(dot(edge2, face_normal)) > 1e-2
        throw(ArgumentError("geometry is not planar"))
    end

    # The `face_normal` computed above might not be exactly orthogonal to the plane
    # spanned by `edge1` and `edge2`, but this is important for some computations later.
    computed_face_normal = SVector(Tuple(normalize(cross(edge1, edge2))))
    computed_face_normal *= sign(dot(computed_face_normal, face_normal))

    return (; face=(face_vertices[:, 1], face_vertices[:, 2], face_vertices[:, 3]),
            face_normal=computed_face_normal)
end

# According to:
# https://logicatcore.github.io/scratchpad/lidar/sensor-fusion/jupyter/2021/04/20/3D-Oriented-Bounding-Box.html
function oriented_bounding_box(point_cloud)
    covariance_matrix = Statistics.cov(point_cloud; dims=2)
    eigen_vectors = Statistics.eigvecs(covariance_matrix)
    means = Statistics.mean(point_cloud, dims=2)

    centered_data = point_cloud .- means

    aligned_coords = eigen_vectors' * centered_data

    min_corner = minimum(aligned_coords, dims=2)
    max_corner = maximum(aligned_coords, dims=2)

    face_vertices = hcat([min_corner[1], max_corner[2], min_corner[3]],
                         min_corner, max_corner)

    return eigen_vectors * face_vertices .+ means
end

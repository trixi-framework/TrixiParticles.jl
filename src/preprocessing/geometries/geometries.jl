include("polygon.jl")
include("triangle_mesh.jl")
include("io.jl")

@inline eachface(mesh) = Base.OneTo(nfaces(mesh))

function Base.setdiff(initial_condition::InitialCondition,
                      geometries::Union{Polygon, TriangleMesh}...)
    geometry = first(geometries)

    if ndims(geometry) != ndims(initial_condition)
        throw(ArgumentError("all passed geometries must have the same dimensionality as the initial condition"))
    end

    coords = reinterpret(reshape, SVector{ndims(geometry), eltype(geometry)},
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

    coords = reinterpret(reshape, SVector{ndims(geometry), eltype(geometry)},
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

    face_vertices, _, _ = oriented_bounding_box(stack(planar_geometry.vertices))

    # Vectors spanning the face
    edge1 = face_vertices[:, 2] - face_vertices[:, 1]
    edge2 = face_vertices[:, 3] - face_vertices[:, 1]

    if abs(dot(edge1, face_normal)) > 1e-2 || abs(dot(edge2, face_normal)) > 1e-2
        throw(ArgumentError("geometry is not planar"))
    end

    return (; face=(face_vertices[:, 1], face_vertices[:, 2], face_vertices[:, 3]),
            face_normal=face_normal)
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

    rect_coords = hcat(min_corner, max_corner,
                       [min_corner[1], max_corner[2], min_corner[3]])

    rotated_rect_coords = eigen_vectors * rect_coords .+ means

    return rotated_rect_coords, eigen_vectors, (min_corner, max_corner)
end

"""
    OrientedBox(; box_origin, orientation_vector, edge_lengths::Tuple)
    OrientedBox(geometry::TrixiParticles.TriangleMesh; scale::Real=1)

Constructor for a parallelogram spanned by two orthogonal edge vectors,
a parallelepiped spanned by three orthogonal edge vectors,
or a parallelepiped enclosing a 3D geometry.

# Arguments
- `geometry`: Geometry returned by [`load_geometry`](@ref).

# Keywords
- `box_origin`: The origin corner of the box.
- `orientation_vector`: A vector describing the main direction of the box.
- `edge_lengths::Tuple`: The lengths of the edges of the box:
                            - In 2D: `(length_x, length_y)`
                            - In 3D: `(length_x, length_y, length_z)`

# Examples
```jldoctest; output=false
# 2D
OrientedBox(
    box_origin = [0.0, 0.0],
    orientation_vector = [1.0, 1.0],
    edge_lengths = (2.0, 1.0)
)

# 3D
OrientedBox(
    box_origin = [0.5, -0.2, 0.0],
    orientation_vector = [0.0, 0.0, 1.0],
    edge_lengths = (1.0, 2.0, 3.0)
)

# output
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ OrientedBox (3D)                                                                                 │
│ ════════════════                                                                                 │
│ zone origin: ……………………………………………… [0.5, -0.2, 0.0]                                                 │
│ edge lengths: …………………………………………… (1.0, 2.0, 3.0)                                                  │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
struct OrientedBox{NDIMS, ELTYPE <: Real, SV}
    box_origin       :: SVector{NDIMS, ELTYPE}
    spanning_vectors :: SV
end

# Constructor with orientation vector (from box_origin to opposite corner)
function OrientedBox(; box_origin, orientation_vector, edge_lengths::Tuple)
    NDIMS = length(box_origin)

    @assert length(orientation_vector) == NDIMS
    @assert length(edge_lengths) == NDIMS

    # Normalize the orientation vector to get the primary direction
    primary_direction = normalize(orientation_vector)

    # Create orthogonal basis vectors
    R = orientation_matrix(primary_direction, edge_lengths)

    spanning_vectors = ntuple(i -> SVector{NDIMS}(R[:, i] * edge_lengths[i]),
                              length(edge_lengths))

    return OrientedBox(SVector{NDIMS}(box_origin), spanning_vectors)
end

function OrientedBox(geometry::TriangleMesh)
    point_cloud = stack(geometry.vertices)
    vertices, eigen_vectors, (min_corner, max_corner) = oriented_bounding_box(point_cloud)

    # Use the first vertex as box origin (bottom-left-front corner)
    box_origin = SVector{3}(vertices[:, 1])

    # Calculate edge lengths from min/max corners
    edge_lengths = max_corner - min_corner

    # Create spanning vectors using the eigen vectors scaled by edge lengths
    spanning_vectors = ntuple(i -> SVector{3}(eigen_vectors[:, i] * edge_lengths[i]), 3)

    return OrientedBox(box_origin, spanning_vectors)
end

@inline Base.ndims(::OrientedBox{NDIMS}) where {NDIMS} = NDIMS

function Base.show(io::IO, ::MIME"text/plain", box::OrientedBox)
    @nospecialize box # reduce precompilation time

    if get(io, :compact, false)
        show(io, box)
    else
        summary_header(io, "OrientedBox ($(ndims(box))D)")
        summary_line(io, "box origin", box.box_origin)
        summary_line(io, "edge lengths", norm.(box.spanning_vectors))
        summary_footer(io)
    end
end

function orientation_matrix(primary_direction, ::NTuple{2})
    perpendicular = [-primary_direction[2], primary_direction[1]]
    return hcat(primary_direction, perpendicular)
end

function orientation_matrix(primary_direction, ::NTuple{3})
    # Find two orthogonal vectors to the primary direction
    # Choose a vector that's not parallel to primary_direction
    temp = abs(primary_direction[1]) < 0.9 ? [1.0, 0.0, 0.0] : [0.0, 1.0, 0.0]

    # Gram-Schmidt orthogonalization
    v2 = temp - dot(temp, primary_direction) * primary_direction
    v2 = normalize(v2)
    v3 = cross(primary_direction, v2)

    return hcat(primary_direction, v2, v3)
end

function is_in_oriented_box(coordinates::AbstractArray, box)
    is_in_box = fill(false, nparticles(initial_condition))
    @threaded default_backend(coordinates) for particle in eachparticle(initial_condition)
        particle_coords = current_coords(coordinates, box, particle)
        is_in_box[particle] = is_in_oriented_box(particle_coords, box)
    end

    return findall(is_in_box)
end

@inline function is_in_oriented_box(particle_coords::SVector{NDIMS}, box) where {NDIMS}
    (; spanning_vectors, zone_origin) = box
    relative_position = particle_coords - zone_origin

    for dim in 1:NDIMS
        span_dim = spanning_vectors[dim]
        # Checks whether the projection of the particle position
        # falls within the range of the zone
        if !(0 <= dot(relative_position, span_dim) <= dot(span_dim, span_dim))

            # Particle is not in box
            return false
        end
    end

    # Particle is in box
    return true
end

@inline function Base.intersect(initial_condition::InitialCondition, boxes::OrientedBox...)
    (; coordinates, density, mass, velocity, pressure, particle_spacing) = initial_condition
    box = first(boxes)

    if ndims(box) != ndims(initial_condition)
        throw(ArgumentError("all passed `OrientedBox`s must have the same dimensionality as the initial condition"))
    end

    keep_indices = fill(false, nparticles(initial_condition))
    @threaded default_backend(coordinates) for particle in eachparticle(initial_condition)
        particle_coords = current_coords(coordinates, initial_condition, particle)

        keep_indices[particle] = is_in_oriented_box(particle_coords, box)
    end

    result = InitialCondition(; coordinates=coordinates[:, keep_indices],
                              density=density[keep_indices], mass=mass[keep_indices],
                              velocity=velocity[:, keep_indices],
                              pressure=pressure[keep_indices],
                              particle_spacing)

    return intersect(result, Base.tail(boxes)...)
end

@inline function Base.setdiff(initial_condition::InitialCondition, boxes::OrientedBox...)
    (; coordinates, density, mass, velocity, pressure, particle_spacing) = initial_condition
    box = first(boxes)

    if ndims(box) != ndims(initial_condition)
        throw(ArgumentError("all passed `OrientedBox`s must have the same dimensionality as the initial condition"))
    end

    keep_indices = fill(false, nparticles(initial_condition))
    @threaded default_backend(coordinates) for particle in eachparticle(initial_condition)
        particle_coords = current_coords(coordinates, initial_condition, particle)

        keep_indices[particle] = !is_in_oriented_box(particle_coords, box)
    end

    result = InitialCondition(; coordinates=coordinates[:, keep_indices],
                              density=density[keep_indices], mass=mass[keep_indices],
                              velocity=velocity[:, keep_indices],
                              pressure=pressure[keep_indices],
                              particle_spacing)

    return setdiff(result, Base.tail(boxes)...)
end

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

# This method is used in `boundary_zone.jl` and is defined here
# to avoid circular dependencies with `TriangleMesh`
function calculate_spanning_vectors(plane::TriangleMesh, zone_width)
    plane_normal = normalize(sum(plane.face_normals) / nfaces(plane))

    plane_points = oriented_bounding_box(stack(plane.vertices))

    # Vectors spanning the plane
    edge1 = plane_points[:, 2] - plane_points[:, 1]
    edge2 = plane_points[:, 3] - plane_points[:, 1]

    if !isapprox(abs.(normalize(cross(edge2, edge1))), abs.(plane_normal), atol=1e-2)
        throw(ArgumentError("`plane` might be not planar"))
    end

    return hcat(plane_normal * zone_width, edge1, edge2), SVector(plane_points[:, 1]...)
end

# According to:
# https://logicatcore.github.io/scratchpad/lidar/sensor-fusion/jupyter/2021/04/20/3D-Oriented-Bounding-Box.html
function oriented_bounding_box(point_cloud)
    covariance_matrix = Statistics.cov(point_cloud; dims=2)
    eigen_vectors = Statistics.eigvecs(covariance_matrix)
    means = Statistics.mean(point_cloud, dims=2)

    centered_data = point_cloud .- means

    aligned_coords = eigen_vectors' * centered_data

    min_corner = SVector(minimum(aligned_coords[1, :]),
                         minimum(aligned_coords[2, :]),
                         minimum(aligned_coords[3, :]))
    max_corner = SVector(maximum(aligned_coords[1, :]),
                         maximum(aligned_coords[2, :]),
                         maximum(aligned_coords[3, :]))

    plane_points = hcat(min_corner, max_corner,
                        [min_corner[1], max_corner[2], min_corner[3]])

    return eigen_vectors * plane_points .+ means
end

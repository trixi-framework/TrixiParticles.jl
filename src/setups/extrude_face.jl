function ExtrudeFace(face_points::NTuple; particle_spacing, direction, n_extrude::Int,
                     velocity=zeros(length(direction)),
                     mass=nothing, density=nothing, pressure=0.0)
    NDIMS = length(direction)
    face_coords, particle_spacing = generate_face_coords(face_points, particle_spacing)
    coordinates_ = (face_coords .+ i * particle_spacing * normalize(direction) for i in 0:n_extrude)

    coordinates = reshape(stack(coordinates_),
                          (NDIMS, size(face_coords, 2) * (n_extrude + 1)))

    return InitialCondition(; coordinates, velocity, density, mass, pressure,
                            particle_spacing)
end

function ExtrudeFace(face_coords::AbstractArray; particle_spacing, direction,
                     n_extrude::Int, velocity=zeros(length(direction)),
                     mass=nothing, density=nothing, pressure=0.0)
    NDIMS = length(direction)

    coordinates_ = (face_coords .+ i * particle_spacing * normalize(direction) for i in 0:n_extrude)

    coordinates = reshape(stack(coordinates_),
                          (NDIMS, size(face_coords, 2) * (n_extrude + 1)))

    return InitialCondition(; coordinates, velocity, density, mass, pressure,
                            particle_spacing)
end

function generate_face_coords(face_points::NTuple{2}, particle_spacing)
    # Verify that points are in 2D space
    if any(length.(face_points) .!= 2)
        throw(ArgumentError("all points must be 2D coordinates"))
    end

    n_points = ceil(Int, norm(face_points[2] - face_points[1]) / particle_spacing)

    coords = stack(range(face_points[1], face_points[2], length=n_points))
    particle_spacing_ = norm(coords[:, 1] - coords[:, 2])

    if round(particle_spacing, digits=4) != round(particle_spacing_, digits=4)
        @info "The desired size ($(norm(face_points[2] - face_points[1])) is not a " *
              "multiple of the particle spacing $particle_spacing." *
              "\nNew particle spacing is set to $particle_spacing_."
    end

    return coords, particle_spacing_
end

function generate_face_coords(face_points::NTuple{3}, particle_spacing)
    # Verify that points are in 3D space
    if any(length.(face_points) .!= 3)
        throw(ArgumentError("all points must be 3D coordinates"))
    end

    # Vectors defining the edges of the parallelogram
    edge1 = face_points[2] - face_points[1]
    edge2 = face_points[3] - face_points[1]

    # Check if the points are collinear
    if norm(cross(edge1, edge2)) == 0
        throw(ArgumentError("the points must not be collinear"))
    end

    # Determine the number of points along each edge
    num_points_edge1 = ceil(Int, norm(edge1) / particle_spacing)
    num_points_edge2 = ceil(Int, norm(edge2) / particle_spacing)

    points_coords = Vector{SVector{3, Float64}}(undef,
                                                (num_points_edge1 + 1) *
                                                (num_points_edge2 + 1))
    index = 1
    for i in 0:num_points_edge1
        for j in 0:num_points_edge2
            point_on_plane = face_points[1] + (i / num_points_edge1) * edge1 +
                             (j / num_points_edge2) * edge2
            points_coords[index] = point_on_plane
            index += 1
        end
    end

    coords = stack(points_coords)
    particle_spacing_ = norm(coords[:, 1] - coords[:, 2])

    if round(particle_spacing, digits=4) != round(particle_spacing_, digits=4)
        @info "The desired size $(norm(face_points[2] - face_points[1])) is not a " *
              "multiple of the particle spacing $particle_spacing." *
              "\nNew particle spacing is set to $particle_spacing_."
    end

    return coords, particle_spacing_
end

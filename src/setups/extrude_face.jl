function ExtrudeFace(face_points; particle_spacing, direction, n_extrude=0,
                     velocity=zeros(length(direction)),
                     mass=nothing, density=nothing, pressure=0.0)
    NDIMS = length(direction)
    face_coords, particle_spacing_ = sample_face(face_points, particle_spacing)

    if round(particle_spacing, digits=4) != round(particle_spacing_, digits=4)
        @info "The desired size is not a multiple of the particle spacing $particle_spacing." *
              "\nNew particle spacing is set to $particle_spacing_."
    end

    coords = (face_coords .+ i * particle_spacing * normalize(direction) for i in 0:n_extrude)

    coords_ = reshape(stack(coords), (NDIMS, size(face_coords, 2) * (n_extrude + 1)))

    return InitialCondition(; coords_, velocity, density, mass, pressure,
                            particle_spacing)
end

function sample_face(face_points::AbstractMatrix, particle_spacing)

    # `face_points` is already a sampled face
    return face_points, particle_spacing
end

function sample_face(face_points, particle_spacing)

    # Convert to tuple
    return sample_face(tuple(face_points...), particle_spacing)
end

function sample_face(face_points::NTuple{2}, particle_spacing)
    # Verify that points are in 2D space
    if any(length.(face_points) .!= 2)
        throw(ArgumentError("all points must be 2D coordinates"))
    end

    n_points = ceil(Int, norm(face_points[2] - face_points[1]) / particle_spacing)

    coords = stack(range(face_points[1], face_points[2], length=n_points))
    particle_spacing_new = norm(coords[:, 1] - coords[:, 2])

    return coords, particle_spacing_new
end

function sample_face(face_points::NTuple{3}, particle_spacing)
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

    coords = zeros(3, (num_points_edge1 + 1) * (num_points_edge2 + 1))

    index = 1
    for i in 0:num_points_edge1
        for j in 0:num_points_edge2
            point_on_plane = face_points[1] + (i / num_points_edge1) * edge1 +
                             (j / num_points_edge2) * edge2
            coords[:, index] = point_on_plane
            index += 1
        end
    end

    particle_spacing_new = norm(coords[:, 1] - coords[:, 2])

    return coords, particle_spacing_new
end

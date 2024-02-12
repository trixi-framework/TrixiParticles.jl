"""
    ExtrudeGeometry(geometry; particle_spacing, direction, n_extrude=0,
                    velocity=zeros(length(direction)),
                    mass=nothing, density=nothing, pressure=0.0)
Extrude either a line, a plane or a shape along a specific direction.

# Arguments
- `geometry`:           Either points defining a 3D plane (2D line)
                        or particle coordinates defining a specific shape.

# Keywords
- `particle_spacing`:   Spacing between the particles.
- `direction`:          Vector defining the extrusion direction.
- `n_extrude=1`         Number of `geometry` layers in extrude direction.
- `velocity`:           Either a function mapping each particle's coordinates to its velocity,
                        or, for a constant fluid velocity, a vector holding this velocity.
                        Velocity is constant zero by default.
- `mass`:               Either `nothing` (default) to automatically compute particle mass from particle
                        density and spacing, or a function mapping each particle's coordinates to its mass,
                        or a scalar for a constant mass over all particles.
- `density`:            Either a function mapping each particle's coordinates to its density,
                        or a scalar for a constant density over all particles.
                        Obligatory when not using a state equation. Cannot be used together with
                        `state_equation`.
- `pressure`:           Scalar to set the pressure of all particles to this value.
                        This is only used by the [`EntropicallyDampedSPHSystem`](@ref) and
                        will be overwritten when using an initial pressure function in the system.
                        Cannot be used together with hydrostatic pressure gradient.
- `tlsph`:              With the [`TotalLagrangianSPHSystem`](@ref), particles need to be placed
                        on the boundary of the shape and not one particle radius away, as for fluids.
                        When `tlsph=true`, particles will be placed on the boundary of the shape.

# Examples
```julia
# 2D
p1 = [0.0, 0.0]
p2 = [1.0, 1.0]

direction = [-1.0, 1.0]

shape = ExtrudeGeometry((p1, p2); direction, particle_spacing=0.1, n_extrude=4, density=1000.0)

# 3D Plane
p1 = [0.0, 0.0, 0.0]
p2 = [0.5, 1.0, 0.0]
p3 = [1.0, 0.2, 0.0]

direction = [0.0, 0.0, 1.0]

shape = ExtrudeGeometry((p1, p2, p3); direction, particle_spacing=0.1, n_extrude=4, density=1000.0)

# Extrude a 2D shape to a 3D shape
shape = SphereShape(0.1, 0.5, (0.2, 0.4), 1000.0, n_layers=3,
                    sphere_type=RoundSphere(end_angle=pi))

direction = [0.0, 0.0, 1.0]

shape = ExtrudeGeometry(shape; direction, particle_spacing=0.1, n_extrude=4, density=1000.0)
```

!!! warning
    `particle_spacing` between extrusion layers may differ from shapes `particle_spacing`.
"""
function ExtrudeGeometry(geometry; particle_spacing, direction, n_extrude=1,
                         velocity=zeros(length(direction)), tlsph=false,
                         mass=nothing, density=nothing, pressure=0.0)
    direction_ = normalize(direction)
    NDIMS = length(direction_)

    geometry = consider_particle_placement(geometry, direction_, particle_spacing, tlsph)

    face_coords, particle_spacing_ = sample_plane(geometry, particle_spacing)

    if round(particle_spacing, digits=4) != round(particle_spacing_, digits=4)
        @info "The desired size is not a multiple of the particle spacing $particle_spacing." *
              "\nNew particle spacing is set to $particle_spacing_."
    end

    coords = (face_coords .+ i * particle_spacing_ * direction_ for i in 0:(n_extrude - 1))

    coordinates = reshape(stack(coords), (NDIMS, size(face_coords, 2) * n_extrude))

    return InitialCondition(; coordinates, velocity, density, mass, pressure,
                            particle_spacing=particle_spacing_)
end

function sample_plane(geometry::AbstractMatrix, particle_spacing)

    # `geometry` is already a sampled shape
    return geometry, particle_spacing
end

function sample_plane(shape::InitialCondition, particle_spacing)
    if ndims(shape) == 2
        # Extruding a 2D shape results in a 3D shape
        coords = vcat(shape.coordinates, zeros(1, size(shape.coordinates, 2)))

        return coords, particle_spacing
    end

    return shape.coordinates, particle_spacing
end

function sample_plane(plane_points, particle_spacing)

    # Convert to tuple
    return sample_plane(tuple(plane_points...), particle_spacing)
end

function sample_plane(plane_points::NTuple{2}, particle_spacing)
    # Verify that points are in 2D space
    if any(length.(plane_points) .!= 2)
        throw(ArgumentError("all points must be 2D coordinates"))
    end

    n_points = ceil(Int, norm(plane_points[2] - plane_points[1]) / particle_spacing) + 1

    coords = stack(range(plane_points[1], plane_points[2], length=n_points))
    particle_spacing_new = norm(coords[:, 1] - coords[:, 2])

    return coords, particle_spacing_new
end

function sample_plane(plane_points::NTuple{3}, particle_spacing)
    # Verify that points are in 3D space
    if any(length.(plane_points) .!= 3)
        throw(ArgumentError("all points must be 3D coordinates"))
    end

    # Vectors defining the edges of the parallelogram
    edge1 = plane_points[2] - plane_points[1]
    edge2 = plane_points[3] - plane_points[1]

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
            point_on_plane = plane_points[1] + (i / num_points_edge1) * edge1 +
                             (j / num_points_edge2) * edge2
            coords[:, index] = point_on_plane
            index += 1
        end
    end

    particle_spacing_new = min(norm(edge1 / num_points_edge1),
                               norm(edge2 / num_points_edge2))

    return coords, particle_spacing_new
end

function consider_particle_placement(geometry::Union{AbstractMatrix, InitialCondition},
                                     direction, particle_spacing, tlsph)
    return geometry
end

function consider_particle_placement(plane_points, direction, particle_spacing, tlsph)
    consider_particle_placement(tuple(plane_points...), direction, particle_spacing, tlsph)
end

function consider_particle_placement(plane_points::NTuple{2}, direction, particle_spacing,
                                     tlsph)
    # With TLSPH, particles need to be AT the min coordinates and not half a particle
    # spacing away from it.
    (tlsph) && (return plane_points)

    plane_point1 = copy(plane_points[1])
    plane_point2 = copy(plane_points[2])

    # Vectors shifting the points in the corresponding direction
    dir1 = 0.5 * particle_spacing * direction
    dir2 = 0.5 * particle_spacing * normalize(plane_point2 - plane_point1)

    plane_point1 .+= dir1 + dir2
    plane_point2 .+= dir1 - dir2

    return (plane_point1, plane_point2)
end

function consider_particle_placement(plane_points::NTuple{3}, direction, particle_spacing,
                                     tlsph)
    # With TLSPH, particles need to be AT the min coordinates and not half a particle
    # spacing away from it.
    (tlsph) && (return plane_points)

    plane_point1 = copy(plane_points[1])
    plane_point2 = copy(plane_points[2])
    plane_point3 = copy(plane_points[3])

    edge1 = normalize(plane_point2 - plane_point1)
    edge2 = normalize(plane_point3 - plane_point1)

    # Vectors shifting the points in the corresponding direction
    dir1 = 0.5 * particle_spacing * direction
    dir2 = 0.5 * particle_spacing * edge1
    dir3 = 0.5 * particle_spacing * edge2

    plane_point1 .+= dir1 + dir2 + dir3
    plane_point2 .+= dir1 - dir2 + dir3
    plane_point3 .+= dir1 + dir2 - dir3

    return (plane_point1, plane_point2, plane_point3)
end

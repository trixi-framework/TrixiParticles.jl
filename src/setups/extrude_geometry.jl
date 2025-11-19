@doc raw"""
    extrude_geometry(geometry; particle_spacing, direction, n_extrude::Integer,
                     velocity=zeros(length(direction)),
                     mass=nothing, density=nothing, pressure=0.0)

Extrude either a line, a plane or a shape along a specific direction.
Returns an [`InitialCondition`](@ref).

# Arguments
- `geometry`:           Either particle coordinates or an [`InitialCondition`](@ref)
                        defining a 2D shape to extrude to a 3D volume, or two 2D points
                        ``(A, B)`` defining the interval ``[A, B]`` to extrude to a plane
                        in 2D, or three 3D points ``(A, B, C)`` defining the parallelogram
                        spanned by the vectors ``\widehat{AB}`` and ``\widehat {AC}`` to extrude
                        to a parallelepiped.

# Keywords
- `particle_spacing`:   Spacing between the particles. Can be omitted when `geometry` is an
                        `InitialCondition` (unless `geometry.particle_spacing == -1`).
- `direction`:          A vector that specifies the direction in which to extrude.
- `n_extrude`:          Number of layers of particles created in the direction of extrusion.
- `velocity`:           Either a function mapping each particle's coordinates to its velocity,
                        or, for a constant fluid velocity, a vector holding this velocity.
                        Velocity is constant zero by default.
- `mass`:               Either `nothing` (default) to automatically compute particle mass from particle
                        density and spacing, or a function mapping each particle's coordinates to its mass,
                        or a scalar for a constant mass over all particles.
- `density`:            Either a function mapping each particle's coordinates to its density,
                        or a scalar for a constant density over all particles.
- `pressure`:           Scalar to set the pressure of all particles to this value.
                        This is only used by the [`EntropicallyDampedSPHSystem`](@ref) and
                        will be overwritten when using an initial pressure function in the system.
- `place_on_shell`:     If `place_on_shell=true`, particles will be placed
                        on the shell of the geometry. For example,
                        the [`TotalLagrangianSPHSystem`](@ref) requires particles to be placed
                        on the shell of the geometry and not half a particle spacing away,
                        as for fluids.

# Examples
```jldoctest; output = false
# Extrude a line in 2D to a plane in 2D
p1 = [0.0, 0.0]
p2 = [1.0, 1.0]

direction = [-1.0, 1.0]

shape = extrude_geometry((p1, p2); direction, particle_spacing=0.1, n_extrude=4, density=1000.0)

# Extrude a parallelogram in 3D to a parallelepiped in 3D
p1 = [0.0, 0.0, 0.0]
p2 = [0.5, 1.0, 0.0]
p3 = [1.0, 0.2, 0.0]

direction = [0.0, 0.0, 1.0]

shape = extrude_geometry((p1, p2, p3); direction, particle_spacing=0.1, n_extrude=4, density=1000.0)

# Extrude a 2D shape (here: a disc) to a 3D shape (here: a cylinder)
shape = SphereShape(0.1, 0.5, (0.2, 0.4), 1000.0, n_layers=3,
                    sphere_type=RoundSphere(end_angle=pi))

direction = [0.0, 0.0, 1.0]

shape = extrude_geometry(shape; direction, particle_spacing=0.1, n_extrude=4, density=1000.0)

# output
┌ Info: The desired line length 1.314213562373095 is not a multiple of the particle spacing 0.1.
└ New line length is set to 1.3.
┌ Info: The desired edge 1 length 1.0180339887498948 is not a multiple of the particle spacing 0.1.
└ New edge 1 length is set to 1.0.
┌ Info: The desired edge 2 length 0.9198039027185568 is not a multiple of the particle spacing 0.1.
└ New edge 2 length is set to 0.9.
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ InitialCondition{Float64}                                                                        │
│ ═════════════════════════                                                                        │
│ #dimensions: ……………………………………………… 3                                                                │
│ #particles: ………………………………………………… 144                                                              │
│ particle spacing: ………………………………… 0.1                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

!!! warning "Experimental Implementation"
    This is an experimental feature and may change in any future releases.
"""
function extrude_geometry(geometry; particle_spacing=-1, direction, n_extrude::Integer,
                          velocity=zeros(length(direction)), place_on_shell=false,
                          mass=nothing, density=nothing, pressure=0.0)
    direction_ = normalize(direction)
    NDIMS = length(direction_)

    if geometry isa InitialCondition && geometry.particle_spacing > 0
        if particle_spacing > 0 && particle_spacing != geometry.particle_spacing
            throw(ArgumentError("`particle_spacing` must be -1 when using an `InitialCondition`"))
        end
        particle_spacing = geometry.particle_spacing
    end

    if particle_spacing <= 0
        throw(ArgumentError("`particle_spacing` must be specified when not extruding an `InitialCondition`"))
    end

    geometry = shift_plane_corners(geometry, direction_, particle_spacing, place_on_shell)

    face_coords = sample_plane(geometry, particle_spacing; place_on_shell=place_on_shell)

    coords = (face_coords .+ i * particle_spacing * direction_ for i in 0:(n_extrude - 1))

    # In this context, `stack` is faster than `hcat(coords...)`
    coordinates = reshape(stack(coords), (NDIMS, size(face_coords, 2) * n_extrude))

    if geometry isa InitialCondition
        density = vcat(geometry.density, (geometry.density for i in 1:(n_extrude - 1))...)
    end

    return InitialCondition(; coordinates, velocity, density, mass, pressure,
                            particle_spacing=particle_spacing)
end

# For corners/endpoints of a plane/line, sample the plane/line with particles.
# For 2D coordinates or an `InitialCondition`, add a third dimension.
function sample_plane(geometry::AbstractMatrix, particle_spacing; place_on_shell)
    if size(geometry, 1) == 2
        # Extruding a 2D shape results in a 3D shape

        # When `place_on_shell=true`, particles will be placed on the x-y plane
        coords = vcat(geometry,
                      fill(place_on_shell ? 0 : particle_spacing / 2, size(geometry, 2))')

        # TODO: 2D shapes not only in x-y plane but in any user-defined plane
        return coords
    end

    return geometry
end

function sample_plane(shape::InitialCondition, particle_spacing; place_on_shell)
    if ndims(shape) == 2
        # Extruding a 2D shape results in a 3D shape

        # When `place_on_shell=true`, particles will be placed on the x-y plane
        coords = vcat(shape.coordinates,
                      fill(place_on_shell ? 0 : particle_spacing / 2,
                           size(shape.coordinates, 2))')

        # TODO: 2D shapes not only in x-y plane but in any user-defined plane
        return coords
    end

    return shape.coordinates
end

function sample_plane(plane_points, particle_spacing; place_on_shell=nothing)

    # Convert to tuple
    return sample_plane(tuple(plane_points...), particle_spacing; place_on_shell=nothing)
end

function sample_plane(plane_points::NTuple{2}, particle_spacing; place_on_shell=nothing)
    # Verify that points are in 2D space
    if any(length.(plane_points) .!= 2)
        throw(ArgumentError("all points must be 2D coordinates"))
    end

    p1 = plane_points[1]
    p2 = plane_points[2]

    line_dir = p2 - p1
    line_length = norm(line_dir)

    n_particles,
    new_length = round_n_particles(line_length, particle_spacing,
                                   "line length")

    coords = stack([p1 + i * particle_spacing * normalize(line_dir) for i in 0:n_particles])

    return coords
end

function sample_plane(plane_points::NTuple{3}, particle_spacing; place_on_shell=nothing)
    # Verify that points are in 3D space
    if any(length.(plane_points) .!= 3)
        throw(ArgumentError("all points must be 3D coordinates"))
    end

    point1_ = SVector{3}(plane_points[1])
    point2_ = SVector{3}(plane_points[2])
    point3_ = SVector{3}(plane_points[3])

    # Vectors defining the edges of the parallelogram
    edge1 = point2_ - point1_
    edge2 = point3_ - point1_

    # Check if the points are collinear
    if isapprox(norm(cross(edge1, edge2)), 0; atol=eps())
        throw(ArgumentError("the vectors `AB` and `AC` must not be collinear"))
    end

    # Determine the number of points along each edge
    num_points_edge1,
    new_length = round_n_particles(norm(edge1), particle_spacing,
                                   "edge 1 length")
    num_points_edge2,
    new_length = round_n_particles(norm(edge2), particle_spacing,
                                   "edge 2 length")

    dir1 = normalize(edge1)
    dir2 = normalize(edge2)
    coords = zeros(3, (num_points_edge1 + 1) * (num_points_edge2 + 1))

    index = 1
    for i in 0:num_points_edge1
        for j in 0:num_points_edge2
            point_on_plane = point1_ + i * particle_spacing * dir1 +
                             j * particle_spacing * dir2
            coords[:, index] = point_on_plane
            index += 1
        end
    end

    return coords
end

# Shift corners of the plane/line inwards by half a particle spacing with `place_on_shell=false`
# because fluid particles need to be half a particle spacing away from the boundary of the shape.
function shift_plane_corners(geometry::Union{AbstractMatrix, InitialCondition},
                             direction, particle_spacing, place_on_shell)
    return geometry
end

function shift_plane_corners(plane_points, direction, particle_spacing, place_on_shell)
    shift_plane_corners(tuple(plane_points...), direction, particle_spacing, place_on_shell)
end

function shift_plane_corners(plane_points::NTuple{2}, direction, particle_spacing,
                             place_on_shell)
    # With `place_on_shell`, particles need to be AT the min coordinates and not half a particle
    # spacing away from it.
    (place_on_shell) && (return plane_points)

    plane_point1 = copy(plane_points[1])
    plane_point2 = copy(plane_points[2])

    # Vectors shifting the points in the corresponding direction
    dir1 = particle_spacing * direction / 2
    dir2 = particle_spacing * normalize(plane_point2 - plane_point1) / 2

    plane_point1 .+= dir1 + dir2
    plane_point2 .+= dir1 - dir2

    return (plane_point1, plane_point2)
end

function shift_plane_corners(plane_points::NTuple{3}, direction, particle_spacing,
                             place_on_shell)
    # With `place_on_shell`, particles need to be AT the min coordinates and not half a particle
    # spacing away from it.
    (place_on_shell) && (return plane_points)

    plane_point1 = copy(plane_points[1])
    plane_point2 = copy(plane_points[2])
    plane_point3 = copy(plane_points[3])

    edge1 = normalize(plane_point2 - plane_point1)
    edge2 = normalize(plane_point3 - plane_point1)

    # Vectors shifting the points in the corresponding direction
    dir1 = particle_spacing * direction / 2
    dir2 = particle_spacing * edge1 / 2
    dir3 = particle_spacing * edge2 / 2

    plane_point1 .+= dir1 + dir2 + dir3
    plane_point2 .+= dir1 - dir2 + dir3
    plane_point3 .+= dir1 + dir2 - dir3

    return (plane_point1, plane_point2, plane_point3)
end

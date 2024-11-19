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
- `tlsph`:              With the [`TotalLagrangianSPHSystem`](@ref), particles need to be placed
                        on the boundary of the shape and not one particle radius away, as for fluids.
                        When `tlsph=true`, particles will be placed on the boundary of the shape.

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
┌ Info: The desired size is not a multiple of the particle spacing 0.1.
└ New particle spacing is set to 0.09387239731236392.
┌ Info: The desired size is not a multiple of the particle spacing 0.1.
└ New particle spacing is set to 0.09198039027185569.
InitialCondition{Float64}(0.1, [0.44999999999999996 0.43096988312782164 … -0.23871756048182058 -0.24999999999999994; 0.4 0.4956708580912724 … 0.5001344202803415 0.4000000000000001; 0.05 0.05 … 0.35000000000000003 0.35000000000000003], [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], [1.0000000000000002, 1.0000000000000002, 1.0000000000000002, 1.0000000000000002, 1.0000000000000002, 1.0000000000000002, 1.0000000000000002, 1.0000000000000002, 1.0000000000000002, 1.0000000000000002  …  1.0000000000000002, 1.0000000000000002, 1.0000000000000002, 1.0000000000000002, 1.0000000000000002, 1.0000000000000002, 1.0000000000000002, 1.0000000000000002, 1.0000000000000002, 1.0000000000000002], [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0  …  1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
```

!!! warning "Experimental Implementation"
    This is an experimental feature and may change in any future releases.
"""
function extrude_geometry(geometry; particle_spacing=-1, direction, n_extrude::Integer,
                          velocity=zeros(length(direction)), tlsph=false,
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

    geometry = shift_plane_corners(geometry, direction_, particle_spacing, tlsph)

    face_coords, particle_spacing_ = sample_plane(geometry, particle_spacing; tlsph=tlsph)

    if !isapprox(particle_spacing, particle_spacing_, rtol=5e-2)
        @info "The desired size is not a multiple of the particle spacing $particle_spacing." *
              "\nNew particle spacing is set to $particle_spacing_."
    end

    coords = (face_coords .+ i * particle_spacing_ * direction_ for i in 0:(n_extrude - 1))

    # In this context, `stack` is faster than `hcat(coords...)`
    coordinates = reshape(stack(coords), (NDIMS, size(face_coords, 2) * n_extrude))

    if geometry isa InitialCondition
        density = vcat(geometry.density, (geometry.density for i in 1:(n_extrude - 1))...)
    end

    return InitialCondition(; coordinates, velocity, density, mass, pressure,
                            particle_spacing=particle_spacing_)
end

# For corners/endpoints of a plane/line, sample the plane/line with particles.
# For 2D coordinates or an `InitialCondition`, add a third dimension.
function sample_plane(geometry::AbstractMatrix, particle_spacing; tlsph)
    if size(geometry, 1) == 2
        # Extruding a 2D shape results in a 3D shape

        # When `tlsph=true`, particles will be placed on the x-y plane
        coords = vcat(geometry, fill(tlsph ? 0.0 : 0.5particle_spacing, size(geometry, 2))')

        # TODO: 2D shapes not only in x-y plane but in any user-defined plane
        return coords, particle_spacing
    end

    return geometry, particle_spacing
end

function sample_plane(shape::InitialCondition, particle_spacing; tlsph)
    if ndims(shape) == 2
        # Extruding a 2D shape results in a 3D shape

        # When `tlsph=true`, particles will be placed on the x-y plane
        coords = vcat(shape.coordinates,
                      fill(tlsph ? 0.0 : 0.5particle_spacing, size(shape.coordinates, 2))')

        # TODO: 2D shapes not only in x-y plane but in any user-defined plane
        return coords, particle_spacing
    end

    return shape.coordinates, particle_spacing
end

function sample_plane(plane_points, particle_spacing; tlsph=nothing)

    # Convert to tuple
    return sample_plane(tuple(plane_points...), particle_spacing; tlsph=nothing)
end

function sample_plane(plane_points::NTuple{2}, particle_spacing; tlsph=nothing)
    # Verify that points are in 2D space
    if any(length.(plane_points) .!= 2)
        throw(ArgumentError("all points must be 2D coordinates"))
    end

    n_points = ceil(Int, norm(plane_points[2] - plane_points[1]) / particle_spacing) + 1

    coords = stack(range(plane_points[1], plane_points[2], length=n_points))
    particle_spacing_new = norm(coords[:, 1] - coords[:, 2])

    return coords, particle_spacing_new
end

function sample_plane(plane_points::NTuple{3}, particle_spacing; tlsph=nothing)
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
    if isapprox(norm(cross(edge1, edge2)), 0.0; atol=eps())
        throw(ArgumentError("the vectors `AB` and `AC` must not be collinear"))
    end

    # Determine the number of points along each edge
    num_points_edge1 = ceil(Int, norm(edge1) / particle_spacing)
    num_points_edge2 = ceil(Int, norm(edge2) / particle_spacing)

    coords = zeros(3, (num_points_edge1 + 1) * (num_points_edge2 + 1))

    index = 1
    for i in 0:num_points_edge1
        for j in 0:num_points_edge2
            point_on_plane = point1_ + (i / num_points_edge1) * edge1 +
                             (j / num_points_edge2) * edge2
            coords[:, index] = point_on_plane
            index += 1
        end
    end

    particle_spacing_new = min(norm(edge1 / num_points_edge1),
                               norm(edge2 / num_points_edge2))

    return coords, particle_spacing_new
end

# Shift corners of the plane/line inwards by half a particle spacing with `tlsph=false`
# because fluid particles need to be half a particle spacing away from the boundary of the shape.
function shift_plane_corners(geometry::Union{AbstractMatrix, InitialCondition},
                             direction, particle_spacing, tlsph)
    return geometry
end

function shift_plane_corners(plane_points, direction, particle_spacing, tlsph)
    shift_plane_corners(tuple(plane_points...), direction, particle_spacing, tlsph)
end

function shift_plane_corners(plane_points::NTuple{2}, direction, particle_spacing, tlsph)
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

function shift_plane_corners(plane_points::NTuple{3}, direction, particle_spacing, tlsph)
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

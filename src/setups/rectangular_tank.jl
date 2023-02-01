@doc raw"""
    RectangularTank(particle_spacing, spacing_ratio, fluid_width, fluid_height,
                    container_width, container_height, fluid_density;
                    n_layers=1, init_velocity=0.0, boundary_density=fluid_density, faces=trues(4))

    RectangularTank(particle_spacing, spacing_ratio,
                    fluid_width, fluid_height, fluid_depth,
                    container_width, container_height, container_depth,
                    fluid_density;
                    n_layers=1, init_velocity=0.0, boundary_density=fluid_density, faces=trues(6))

Rectangular tank filled with a fluid to set up dam-break-style simulations.

# Arguments
- `particle_spacing`:                    Spacing betweeen the fluid particles
- `spacing_ratio`:                       Ratio of `particle_spacing` to boundary particle spacing.
- `fluid_width`, `fluid_height`:         Initial width and height of the fluid system, respectively.
- `container_width`, `container_height`: Initial width and height of the container, respectively.
- `fluid_density`:                       The rest density of the fluid.
A 3D tank is generated by calling the function additonally with the container and fluid depth (see examples below).

# Keywords
- `n_layers`:           Number of boundary layers.
- `init_velocity`:      Initial velocity of the fluid particles.
- `boundary_density`:   Density of the boundary particles (by default set to the rest density)
- `faces`:              By default all faces are generated. Set faces by passing an bit-array of length 4 (2D) or 6 (3D) to generate the faces in the normal direction: -x,+x,-y,+y,-z,+z

# Fields
- `particle_coordinates::Matrix`: Coordinates of the fluid particles
- `particle_velocities::Matrix`: Velocity of the fluid particles
- `particle_masses::Vector`: Masses of the fluid particles
- `particle_densities::Vector`: Densities of the fluid particles
- `boundary_coordinates::Matrix`: Coordinates of the boundary particles
- `boundary_masses::Vector`: Masses of the boundary particles

# Examples
2D:
```julia
setup = RectangularTank(particle_spacing, 3, water_width, water_height,
                        container_width, container_height, particle_density, n_layers=2)
```

3D:
```julia
setup = RectangularTank(particle_spacing, 3, water_width, water_height, water_depth,
                        container_width, container_height, container_depth, particle_density, n_layers=2)
```

See also: [`reset_wall!`](@ref)
"""
struct RectangularTank{NDIMS, NDIMSt2, ELTYPE <: Real}
    particle_coordinates      :: Array{ELTYPE, 2}
    particle_velocities       :: Array{ELTYPE, 2}
    particle_densities        :: Vector{ELTYPE}
    particle_masses           :: Vector{ELTYPE}
    particle_radius           :: Vector{ELTYPE}
    boundary_coordinates      :: Array{ELTYPE, 2}
    boundary_masses           :: Vector{ELTYPE}
    faces_                    :: NTuple{NDIMSt2, Bool} # store if face in dir exists (-x +x -y +y -z +z)
    face_indices              :: NTuple{NDIMSt2, Array{Int, 2}} # see `reset_wall!`
    particle_spacing          :: ELTYPE
    spacing_ratio             :: ELTYPE
    n_layers                  :: Int
    n_particles_per_dimension :: NTuple{NDIMS, Int}

    function RectangularTank(particle_spacing, spacing_ratio, fluid_width, fluid_height,
                             container_width, container_height, fluid_density;
                             n_layers=1, init_velocity=0.0, boundary_density=fluid_density,
                             faces=Tuple(trues(4)))
        NDIMS = 2
        ELTYPE = eltype(particle_spacing)

        # Leave space for the fluid particles
        container_width += particle_spacing
        container_height += particle_spacing

        # Boundary particle data
        n_boundaries_x, n_boundaries_y, container_width,
        container_height = get_boundary_particles_per_dimension(container_width,
                                                                container_height,
                                                                particle_spacing,
                                                                spacing_ratio)

        boundary_coordinates,
        face_indices = initialize_boundaries(particle_spacing / spacing_ratio,
                                             container_width,
                                             container_height,
                                             n_boundaries_x,
                                             n_boundaries_y,
                                             n_layers, faces)
        boundary_masses = boundary_density * (particle_spacing / spacing_ratio)^2 *
                          ones(ELTYPE, size(boundary_coordinates, 2))

        # Particle data
        n_particles_x = get_fluid_particles_per_dimension(fluid_width, particle_spacing,
                                                          "fluid width")
        n_particles_y = get_fluid_particles_per_dimension(fluid_height, particle_spacing,
                                                          "fluid height")

        if container_width < fluid_width - 1e-5 * particle_spacing
            n_particles_x -= 1
            @info "The fluid was overlapping.\n New fluid width is set to $((n_particles_x + 1) * particle_spacing)"
        end

        if container_height < fluid_height - 1e-5 * particle_spacing
            n_particles_y -= 1
            @info "The fluid was overlapping.\n New fluid height is set to $((n_particles_y + 1) * particle_spacing)"
        end

        n_particles_per_dimension = (n_particles_x, n_particles_y)
        particle_coordinates = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
        particle_velocities = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))

        initialize_particles!(particle_coordinates, particle_velocities, particle_spacing,
                              init_velocity, n_particles_per_dimension)
        particle_densities = fluid_density * ones(Float64, prod(n_particles_per_dimension))
        mass = fluid_density * particle_spacing^2
        particle_masses = mass * ones(ELTYPE, prod(n_particles_per_dimension))
        particle_radius = particle_spacing * ones(ELTYPE, prod(n_particles_per_dimension))

        return new{NDIMS, 2 * NDIMS, ELTYPE}(particle_coordinates, particle_velocities,
                                             particle_densities, particle_masses,
                                             particle_radius,
                                             boundary_coordinates, boundary_masses, faces,
                                             face_indices,
                                             particle_spacing, spacing_ratio, n_layers,
                                             n_particles_per_dimension)
    end

    function RectangularTank(particle_spacing, spacing_ratio,
                             fluid_width, fluid_height, fluid_depth,
                             container_width, container_height, container_depth,
                             fluid_density;
                             n_layers=1, init_velocity=0.0, boundary_density=fluid_density,
                             faces=Tuple(trues(6)))
        NDIMS = 3
        ELTYPE = eltype(particle_spacing)
        mass = fluid_density * particle_spacing^3

        # Leave space for the fluid particles
        container_width += particle_spacing
        container_height += particle_spacing
        container_depth += particle_spacing

        # Boundary particle data
        n_boundaries_x, n_boundaries_y, n_boundaries_z, container_width, container_height,
        container_depth = get_boundary_particles_per_dimension(container_width,
                                                               container_height,
                                                               container_depth,
                                                               particle_spacing,
                                                               spacing_ratio)

        boundary_coordinates,
        face_indices = initialize_boundaries(particle_spacing / spacing_ratio,
                                             container_width,
                                             container_height,
                                             container_depth,
                                             n_boundaries_x,
                                             n_boundaries_y,
                                             n_boundaries_z,
                                             n_layers, faces)
        boundary_masses = boundary_density * (particle_spacing / spacing_ratio)^3 *
                          ones(ELTYPE, size(boundary_coordinates, 2))

        # Particle data
        n_particles_x = get_fluid_particles_per_dimension(fluid_width, particle_spacing,
                                                          "fluid width")
        n_particles_y = get_fluid_particles_per_dimension(fluid_height, particle_spacing,
                                                          "fluid height")
        n_particles_z = get_fluid_particles_per_dimension(fluid_depth, particle_spacing,
                                                          "fluid depth")

        if container_width < fluid_width - 1e-5 * particle_spacing
            n_particles_x -= 1
            @info "The fluid was overlapping.\n New fluid width is set to $((n_particles_x + 1) * particle_spacing)"
        end

        if container_height < fluid_height - 1e-5 * particle_spacing
            n_particles_y -= 1
            @info "The fluid was overlapping.\n New fluid height is set to $((n_particles_y + 1) * particle_spacing)"
        end

        if container_depth < fluid_depth - 1e-5 * particle_spacing
            n_particles_z -= 1
            @info "The fluid was overlapping.\n New fluid depth is set to $((n_particles_z + 1) * particle_spacing)"
        end

        n_particles_per_dimension = (n_particles_x, n_particles_y, n_particles_z)

        particle_coordinates = Array{Float64, 2}(undef, 3, prod(n_particles_per_dimension))
        particle_velocities = Array{Float64, 2}(undef, 3, prod(n_particles_per_dimension))

        initialize_particles!(particle_coordinates, particle_velocities, particle_spacing,
                              init_velocity, n_particles_per_dimension)
        particle_densities = fluid_density * ones(Float64, prod(n_particles_per_dimension))
        particle_masses = mass * ones(ELTYPE, prod(n_particles_per_dimension))
        particle_radius = particle_spacing * ones(ELTYPE, prod(n_particles_per_dimension))

        return new{NDIMS, 2 * NDIMS, ELTYPE}(particle_coordinates, particle_velocities,
                                             particle_densities, particle_masses,
                                             particle_radius,
                                             boundary_coordinates, boundary_masses, faces,
                                             face_indices,
                                             particle_spacing, spacing_ratio, n_layers,
                                             n_particles_per_dimension)
    end
end

function initialize_particles!(particle_coordinates, particle_velocities, particle_spacing,
                               init_velocity, n_particles_per_dimension::NTuple{2})
    for y in 1:n_particles_per_dimension[2],
        x in 1:n_particles_per_dimension[1]

        particle = (x - 1) * n_particles_per_dimension[2] + y

        # Coordinates
        particle_coordinates[1, particle] = x * particle_spacing
        particle_coordinates[2, particle] = y * particle_spacing

        # Velocities
        particle_velocities[1, particle] = init_velocity
        particle_velocities[2, particle] = init_velocity
    end
end

function initialize_particles!(particle_coordinates, particle_velocities, particle_spacing,
                               init_velocity, n_particles_per_dimension::NTuple{3})
    for z in 1:n_particles_per_dimension[3],
        y in 1:n_particles_per_dimension[2],
        x in 1:n_particles_per_dimension[1]

        particle = (x - 1) * n_particles_per_dimension[2] * n_particles_per_dimension[3] +
                   (y - 1) * n_particles_per_dimension[3] + z

        # Coordinates
        particle_coordinates[1, particle] = x * particle_spacing
        particle_coordinates[2, particle] = y * particle_spacing
        particle_coordinates[3, particle] = z * particle_spacing

        # Velocities
        particle_velocities[1, particle] = init_velocity
        particle_velocities[2, particle] = init_velocity
        particle_velocities[3, particle] = init_velocity
    end
end

# 2D
function initialize_boundaries(particle_spacing,
                               container_width, container_height,
                               n_particles_x, n_particles_y, n_layers, faces)
    # Store each particle index
    face_indices_1 = Array{Int, 2}(undef, n_layers, n_particles_y)
    face_indices_2 = Array{Int, 2}(undef, n_layers, n_particles_y)
    face_indices_3 = Array{Int, 2}(undef, n_layers, n_particles_x)
    face_indices_4 = Array{Int, 2}(undef, n_layers, n_particles_x)

    # Create empty array to extend later depending on faces and corners to build
    boundary_coordinates = Array{Float64, 2}(undef, 2, 0)

    # Counts the global index of the particles
    index = 0

    # For odd faces we need to shift the face outwards if we have multiple layers
    layer_offset = -(n_layers - 1) * particle_spacing

    #### Left boundary
    if faces[1]
        left_boundary = RectangularShape(particle_spacing,
                                         (n_layers, n_particles_y),
                                         (layer_offset, particle_spacing))

        # store coordinates of left boundary
        boundary_coordinates = hcat(boundary_coordinates, left_boundary.coordinates)

        # store the indices of each particle
        particles_per_layer = left_boundary.n_particles_per_dimension[2]
        for i in 1:n_layers
            face_indices_1[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    #### Right boundary
    if faces[2]
        right_boundary = RectangularShape(particle_spacing,
                                          (n_layers, n_particles_y),
                                          (container_width, particle_spacing))

        # store coordinates of left boundary
        boundary_coordinates = hcat(boundary_coordinates, right_boundary.coordinates)

        # store the indices of each particle
        particles_per_layer = right_boundary.n_particles_per_dimension[2]
        for i in 1:n_layers
            face_indices_2[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    #### Bottom boundary
    if faces[3]
        bottom_boundary = RectangularShape(particle_spacing,
                                           (n_particles_x, n_layers),
                                           (particle_spacing, layer_offset),
                                           loop_order=:y_first)

        # store coordinates of left boundary
        boundary_coordinates = hcat(boundary_coordinates, bottom_boundary.coordinates)

        # store the indices of each particle
        particles_per_layer = bottom_boundary.n_particles_per_dimension[1]
        for i in 1:n_layers
            face_indices_3[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    #### Top boundary
    if faces[4]
        top_boundary = RectangularShape(particle_spacing,
                                        (n_particles_x, n_layers),
                                        (particle_spacing, container_height),
                                        loop_order=:y_first)

        # store coordinates of left boundary
        boundary_coordinates = hcat(boundary_coordinates, top_boundary.coordinates)

        # store the indices of each particle
        particles_per_layer = top_boundary.n_particles_per_dimension[1]
        for i in 1:n_layers
            face_indices_4[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    #### Add corners
    # Bottom left
    if faces[1] && faces[3]
        bottom_left_corner = RectangularShape(particle_spacing,
                                              (n_layers, n_layers),
                                              (layer_offset, layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, bottom_left_corner.coordinates)
    end

    # Top left
    if faces[1] && faces[4]
        top_left_corner = RectangularShape(particle_spacing,
                                           (n_layers, n_layers),
                                           (layer_offset, container_height))
        boundary_coordinates = hcat(boundary_coordinates, top_left_corner.coordinates)
    end

    # Bottom right
    if faces[2] && faces[3]
        bottom_right_corner = RectangularShape(particle_spacing,
                                               (n_layers, n_layers),
                                               (container_width, layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, bottom_right_corner.coordinates)
    end

    # Top right
    if faces[2] && faces[4]
        top_right_corner = RectangularShape(particle_spacing,
                                            (n_layers, n_layers),
                                            (container_width, container_height))
        boundary_coordinates = hcat(boundary_coordinates, top_right_corner.coordinates)
    end

    return boundary_coordinates,
           (face_indices_1, face_indices_2, face_indices_3, face_indices_4)
end

# 3D
function initialize_boundaries(particle_spacing,
                               container_width, container_height, container_depth,
                               n_particles_x, n_particles_y, n_particles_z,
                               n_layers, faces)
    # Store each particle index
    face_indices_1 = Array{Int, 2}(undef, n_layers, n_particles_y * n_particles_z)
    face_indices_2 = Array{Int, 2}(undef, n_layers, n_particles_y * n_particles_z)
    face_indices_3 = Array{Int, 2}(undef, n_layers, n_particles_x * n_particles_z)
    face_indices_4 = Array{Int, 2}(undef, n_layers, n_particles_x * n_particles_z)
    face_indices_5 = Array{Int, 2}(undef, n_layers, n_particles_x * n_particles_y)
    face_indices_6 = Array{Int, 2}(undef, n_layers, n_particles_x * n_particles_y)

    # Create empty array to extend later depending on faces and corners to build
    boundary_coordinates = Array{Float64, 2}(undef, 3, 0)

    # Counts the global index of the particles
    index = 0

    # For odd faces we need to shift the face outwards if we have multiple layers
    layer_offset = -(n_layers - 1) * particle_spacing

    #### -x boundary (y-z-plane)
    if faces[1]
        x_neg_boundary = RectangularShape(particle_spacing,
                                          (n_layers, n_particles_y, n_particles_z),
                                          (layer_offset, particle_spacing,
                                           particle_spacing))

        # store coordinates of left boundary
        boundary_coordinates = hcat(boundary_coordinates, x_neg_boundary.coordinates)

        # store the indices of each particle
        particles_per_layer = prod(x_neg_boundary.n_particles_per_dimension[2:3])
        for i in 1:n_layers
            face_indices_1[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    #### +x boundary (y-z-plane)
    if faces[2]
        x_pos_boundary = RectangularShape(particle_spacing,
                                          (n_layers, n_particles_y, n_particles_z),
                                          (container_width, particle_spacing,
                                           particle_spacing))

        # store coordinates of left boundary
        boundary_coordinates = hcat(boundary_coordinates, x_pos_boundary.coordinates)

        # store the indices of each particle
        particles_per_layer = prod(x_pos_boundary.n_particles_per_dimension[2:3])
        for i in 1:n_layers
            face_indices_2[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    #### -y boundary (x-z-plane)
    if faces[3]
        y_neg_boundary = RectangularShape(particle_spacing,
                                          (n_particles_x, n_layers, n_particles_z),
                                          (particle_spacing, layer_offset,
                                           particle_spacing),
                                          loop_order=:y_first)

        # store coordinates of left boundary
        boundary_coordinates = hcat(boundary_coordinates, y_neg_boundary.coordinates)

        # store the indices of each particle
        particles_per_layer = prod(y_neg_boundary.n_particles_per_dimension[1:2:3])
        for i in 1:n_layers
            face_indices_3[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    #### +y boundary (x-z-plane)
    if faces[4]
        y_pos_boundary = RectangularShape(particle_spacing,
                                          (n_particles_x, n_layers, n_particles_z),
                                          (particle_spacing, container_height,
                                           particle_spacing), loop_order=:y_first)

        # store coordinates of left boundary
        boundary_coordinates = hcat(boundary_coordinates, y_pos_boundary.coordinates)

        # store the indices of each particle
        particles_per_layer = prod(y_pos_boundary.n_particles_per_dimension[1:2:3])
        for i in 1:n_layers
            face_indices_4[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    #### -z boundary (x-y-plane).
    if faces[5]
        z_neg_boundary = RectangularShape(particle_spacing,
                                          (n_particles_x, n_particles_y, n_layers),
                                          (particle_spacing, particle_spacing,
                                           layer_offset), loop_order=:z_first)

        # store coordinates of left boundary
        boundary_coordinates = hcat(boundary_coordinates, z_neg_boundary.coordinates)

        # store the indices of each particle
        particles_per_layer = prod(z_neg_boundary.n_particles_per_dimension[1:2])
        for i in 1:n_layers
            face_indices_5[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    #### +z boundary (x-y-plane)
    if faces[6]
        z_pos_boundary = RectangularShape(particle_spacing,
                                          (n_particles_x, n_particles_y, n_layers),
                                          (particle_spacing, particle_spacing,
                                           container_depth), loop_order=:z_first)

        # store coordinates of left boundary
        boundary_coordinates = hcat(boundary_coordinates, z_pos_boundary.coordinates)

        # store the indices of each particle
        particles_per_layer = prod(z_pos_boundary.n_particles_per_dimension[1:2])
        for i in 1:n_layers
            face_indices_6[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    #### Add edges
    if faces[1] && faces[3]
        edge_1_3 = RectangularShape(particle_spacing,
                                    (n_layers, n_layers, n_particles_z),
                                    (layer_offset, layer_offset, particle_spacing))
        boundary_coordinates = hcat(boundary_coordinates, edge_1_3.coordinates)
    end

    if faces[1] && faces[4]
        edge_1_4 = RectangularShape(particle_spacing,
                                    (n_layers, n_layers, n_particles_z),
                                    (layer_offset, container_height, particle_spacing))
        boundary_coordinates = hcat(boundary_coordinates, edge_1_4.coordinates)
    end

    if faces[2] && faces[3]
        edge_2_3 = RectangularShape(particle_spacing,
                                    (n_layers, n_layers, n_particles_z),
                                    (container_width, layer_offset, particle_spacing))
        boundary_coordinates = hcat(boundary_coordinates, edge_2_3.coordinates)
    end

    if faces[2] && faces[4]
        edge_2_4 = RectangularShape(particle_spacing,
                                    (n_layers, n_layers, n_particles_z),
                                    (container_width, container_height, particle_spacing))
        boundary_coordinates = hcat(boundary_coordinates, edge_2_4.coordinates)
    end

    if faces[5] && faces[3]
        edge_5_3 = RectangularShape(particle_spacing,
                                    (n_particles_x, n_layers, n_layers),
                                    (particle_spacing, layer_offset, layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, edge_5_3.coordinates)
    end

    if faces[5] && faces[4]
        edge_5_4 = RectangularShape(particle_spacing,
                                    (n_particles_x, n_layers, n_layers),
                                    (particle_spacing, container_height, layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, edge_5_4.coordinates)
    end

    if faces[6] && faces[3]
        edge_6_3 = RectangularShape(particle_spacing,
                                    (n_particles_x, n_layers, n_layers),
                                    (particle_spacing, layer_offset, container_depth))
        boundary_coordinates = hcat(boundary_coordinates, edge_6_3.coordinates)
    end

    if faces[6] && faces[4]
        edge_6_4 = RectangularShape(particle_spacing,
                                    (n_particles_x, n_layers, n_layers),
                                    (particle_spacing, container_height, container_depth))
        boundary_coordinates = hcat(boundary_coordinates, edge_6_4.coordinates)
    end

    if faces[1] && faces[5]
        edge_1_5 = RectangularShape(particle_spacing,
                                    (n_layers, n_particles_y, n_layers),
                                    (layer_offset, particle_spacing, layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, edge_1_5.coordinates)
    end

    if faces[1] && faces[6]
        edge_1_6 = RectangularShape(particle_spacing,
                                    (n_layers, n_particles_y, n_layers),
                                    (layer_offset, particle_spacing, container_depth))
        boundary_coordinates = hcat(boundary_coordinates, edge_1_6.coordinates)
    end

    if faces[5] && faces[2]
        edge_5_2 = RectangularShape(particle_spacing,
                                    (n_layers, n_particles_y, n_layers),
                                    (container_width, particle_spacing, layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, edge_5_2.coordinates)
    end

    if faces[6] && faces[2]
        edge_6_2 = RectangularShape(particle_spacing,
                                    (n_layers, n_particles_y, n_layers),
                                    (container_width, particle_spacing, container_depth))
        boundary_coordinates = hcat(boundary_coordinates, edge_6_2.coordinates)
    end

    #### Add corners
    if faces[1] && faces[3] && faces[5]
        corner_1_3_5 = RectangularShape(particle_spacing,
                                        (n_layers, n_layers, n_layers),
                                        (layer_offset, layer_offset, layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, corner_1_3_5.coordinates)
    end

    if faces[1] && faces[4] && faces[5]
        corner_1_4_5 = RectangularShape(particle_spacing,
                                        (n_layers, n_layers, n_layers),
                                        (layer_offset, container_height, layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, corner_1_4_5.coordinates)
    end

    if faces[1] && faces[3] && faces[6]
        corner_1_3_6 = RectangularShape(particle_spacing,
                                        (n_layers, n_layers, n_layers),
                                        (layer_offset, layer_offset, container_depth))
        boundary_coordinates = hcat(boundary_coordinates, corner_1_3_6.coordinates)
    end

    if faces[1] && faces[4] && faces[6]
        corner_1_4_6 = RectangularShape(particle_spacing,
                                        (n_layers, n_layers, n_layers),
                                        (layer_offset, container_height, container_depth))
        boundary_coordinates = hcat(boundary_coordinates, corner_1_4_6.coordinates)
    end

    if faces[2] && faces[3] && faces[5]
        corner_2_3_5 = RectangularShape(particle_spacing,
                                        (n_layers, n_layers, n_layers),
                                        (container_width, layer_offset, layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, corner_2_3_5.coordinates)
    end

    if faces[2] && faces[4] && faces[5]
        corner_2_4_5 = RectangularShape(particle_spacing,
                                        (n_layers, n_layers, n_layers),
                                        (container_width, container_height, layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, corner_2_4_5.coordinates)
    end

    if faces[2] && faces[3] && faces[6]
        corner_2_3_6 = RectangularShape(particle_spacing,
                                        (n_layers, n_layers, n_layers),
                                        (container_width, layer_offset, container_depth))
        boundary_coordinates = hcat(boundary_coordinates, corner_2_3_6.coordinates)
    end

    if faces[2] && faces[4] && faces[6]
        corner_2_4_6 = RectangularShape(particle_spacing,
                                        (n_layers, n_layers, n_layers),
                                        (container_width, container_height,
                                         container_depth))
        boundary_coordinates = hcat(boundary_coordinates, corner_2_4_6.coordinates)
    end

    return boundary_coordinates,
           (face_indices_1, face_indices_2, face_indices_3, face_indices_4, face_indices_5,
            face_indices_6)
end

@doc raw"""
    reset_wall!(rectangular_tank::RectangularTank, reset_faces, positions)

The selected walls of the tank will be placed at the new positions.

# Arguments
- `reset_faces`: Boolean tuple of 4 (in 2D) or 6 (in 3D) dimensions, similar to `faces` in [`RectangularTank`](@ref).
- `positions`: Tuple of new positions

!!! warning "Warning"
    There are overlapping particles when adjacent walls are moved inwards simultaneously.
"""
function reset_wall!(rectangular_tank, reset_faces, positions)
    @unpack boundary_coordinates, particle_spacing, spacing_ratio,
    n_layers, face_indices = rectangular_tank

    for face in eachindex(reset_faces)
        dim = div(face - 1, 2) + 1

        reset_faces[face] && for layer in 1:n_layers

            # `face_indices` contains the associated particle indices for each face.
            for particle in view(face_indices[face], layer, :)

                # For "odd" faces the layer direction is outwards
                # and for "even" faces inwards.
                layer_shift = if iseven(face)
                    (layer - 1) * particle_spacing / spacing_ratio
                else
                    -(layer - 1) * particle_spacing / spacing_ratio
                end

                # set position
                boundary_coordinates[dim, particle] = positions[face] + layer_shift
            end
        end
    end

    return rectangular_tank
end

function get_fluid_particles_per_dimension(size, spacing, dimension)
    n_particles = round(Int, size / spacing)

    new_size = n_particles * spacing
    if round(new_size, digits=4) != round(size, digits=4)
        print_warn_message(dimension, size, new_size)
    end

    return n_particles
end

function get_boundary_particles_per_dimension(container_width, container_height,
                                              particle_spacing, spacing_ratio)
    n_boundaries_x = round(Int, (container_width / particle_spacing * spacing_ratio))
    n_boundaries_y = round(Int, (container_height / particle_spacing * spacing_ratio))

    new_container_width = n_boundaries_x * (particle_spacing / spacing_ratio)
    new_container_height = n_boundaries_y * (particle_spacing / spacing_ratio)

    if round(new_container_width, digits=4) != round(container_width, digits=4)
        print_warn_message("container width", container_width, new_container_width)
    end
    if round(new_container_height, digits=4) != round(container_height, digits=4)
        print_warn_message("container height", container_height, new_container_height)
    end

    # The container size is larger than the fluid area by one particle spacing,
    # since the boundary particles enclose the fluid particles.
    # For the boundary faces we need the size of the fluid area.
    # Thus, remove one particle again.
    n_boundaries_x -= 1
    n_boundaries_y -= 1

    return n_boundaries_x, n_boundaries_y, new_container_width, new_container_height
end

function get_boundary_particles_per_dimension(container_width, container_height,
                                              container_depth,
                                              particle_spacing, spacing_ratio)
    n_boundaries_x = round(Int, container_width / particle_spacing * spacing_ratio)
    n_boundaries_y = round(Int, container_height / particle_spacing * spacing_ratio)
    n_boundaries_z = round(Int, container_depth / particle_spacing * spacing_ratio)

    new_container_width = n_boundaries_x * (particle_spacing / spacing_ratio)
    new_container_height = n_boundaries_y * (particle_spacing / spacing_ratio)
    new_container_depth = n_boundaries_z * (particle_spacing / spacing_ratio)

    if round(new_container_width, digits=4) != round(container_width, digits=4)
        print_warn_message("container width", container_width, new_container_width)
    end
    if round(new_container_height, digits=4) != round(container_height, digits=4)
        print_warn_message("container height", container_height, new_container_height)
    end
    if round(new_container_depth, digits=4) != round(container_depth, digits=4)
        print_warn_message("container depth", container_depth, new_container_depth)
    end

    # The container size is larger than the fluid area by one particle spacing,
    # since the boundary particles enclose the fluid particles.
    # For the boundary faces we need the size of the fluid area.
    # Thus, remove one particle again.
    n_boundaries_x -= 1
    n_boundaries_y -= 1
    n_boundaries_z -= 1

    return n_boundaries_x, n_boundaries_y, n_boundaries_z,
           new_container_width, new_container_height, new_container_depth
end

function print_warn_message(dimension, size, new_size)
    @info "The desired $dimension $size is not a multiple of the particle spacing.\n New $dimension is set to $new_size."
end

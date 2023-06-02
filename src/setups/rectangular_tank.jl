@doc raw"""
    RectangularTank(particle_spacing, fluid_size::NTuple{2}, tank_size::NTuple{2},
                    fluid_density;
                    n_layers=1, spacing_ratio=1.0, init_velocity=(0.0, 0.0),
                    boundary_density=fluid_density, faces=Tuple(trues(4)))

    RectangularTank(particle_spacing, fluid_size::NTuple{3}, tank_size::NTuple{3},
                    fluid_density;
                    n_layers=1, spacing_ratio=1.0, init_velocity=(0.0, 0.0, 0.0),
                    boundary_density=fluid_density, faces=Tuple(trues(6)))

Rectangular tank filled with a fluid to set up dam-break-style simulations.

# Arguments
- `particle_spacing`:   Spacing between the fluid particles
- `fluid_size`:         The dimensions of the fluid as `(x, y)` (or `(x, y, z)` in 3D).
- `tank_size`:          The dimensions of the tank as `(x, y)` (or `(x, y, z)` in 3D).
- `fluid_density`:      The rest density of the fluid.

# Keywords
- `n_layers`:           Number of boundary layers.
- `spacing_ratio`:      Ratio of `particle_spacing` to boundary particle spacing. A value of 2 means that the boundary particle spacing will be half the fluid particle spacing.
- `init_velocity`:      The initial velocity of the fluid particles as `(x, y)` (or `(x, y, z)` in 3D).
- `boundary_density`:   Density of the boundary particles (by default set to the rest density)
- `faces`:              By default all faces are generated. Set faces by passing an bit-array of length 4 (2D) or 6 (3D) to generate the faces in the normal direction: -x,+x,-y,+y,-z,+z

# Fields
- `coordinates::Matrix`: Coordinates of the fluid particles
- `velocities::Matrix`: Velocity of the fluid particles
- `masses::Vector`: Masses of the fluid particles
- `densities::Vector`: Densities of the fluid particles
- `boundary_coordinates::Matrix`: Coordinates of the boundary particles
- `boundary_masses::Vector`: Masses of the boundary particles
- `boundary_densities::Vector`: Densities of the boundary particles

# Examples
2D:
```julia
setup = RectangularTank(particle_spacing, (water_width, water_height),
                        (container_width, container_height), particle_density,
                        n_layers=2, spacing_ratio=3)
```

3D:
```julia
setup = RectangularTank(particle_spacing, (water_width, water_height, water_depth),
                        (container_width, container_height, container_depth), particle_density, n_layers=2)
```

See also: [`reset_wall!`](@ref)
"""
struct RectangularTank{NDIMS, NDIMSt2, ELTYPE <: Real}
    fluid                     :: InitialCondition{ELTYPE}
    boundary                  :: InitialCondition{ELTYPE}
    faces_                    :: NTuple{NDIMSt2, Bool} # store if face in dir exists (-x +x -y +y -z +z)
    face_indices              :: NTuple{NDIMSt2, Array{Int, 2}} # see `reset_wall!`
    particle_spacing          :: ELTYPE
    spacing_ratio             :: ELTYPE
    n_layers                  :: Int
    n_particles_per_dimension :: NTuple{NDIMS, Int}

    function RectangularTank(particle_spacing, fluid_size::NTuple{2}, tank_size,
                             fluid_density; pressure=[],
                             n_layers=1, spacing_ratio=1.0, init_velocity=(0.0, 0.0),
                             boundary_density=fluid_density, faces=Tuple(trues(4)))
        if particle_spacing < eps()
            throw(ArgumentError("Particle spacing needs to be positive and larger than $(eps())."))
        end

        if fluid_density < eps()
            throw(ArgumentError("Density needs to be positive and larger than $(eps())."))
        end

        NDIMS = 2

        if length(tank_size) != NDIMS
            throw(ArgumentError("`tank_size` must be of length $NDIMS for a $(NDIMS)D problem"))
        end

        ELTYPE = eltype(particle_spacing)

        # Leave space for the fluid particles
        tank_size = tank_size .+ particle_spacing

        # Boundary particle data
        n_boundaries_per_dim, tank_size = boundary_particles_per_dimension(tank_size,
                                                                           particle_spacing,
                                                                           spacing_ratio)

        boundary_coordinates, face_indices = initialize_boundaries(particle_spacing /
                                                                   spacing_ratio, tank_size,
                                                                   n_boundaries_per_dim,
                                                                   n_layers, faces)

        boundary_masses = boundary_density * (particle_spacing / spacing_ratio)^2 *
                          ones(ELTYPE, size(boundary_coordinates, 2))
        boundary_densities = boundary_density * ones(ELTYPE, size(boundary_coordinates, 2))
        boundary_velocities = zeros(ELTYPE, size(boundary_coordinates))

        # Particle data
        n_particles_x = fluid_particles_per_dimension(fluid_size[1], particle_spacing,
                                                      "fluid width")
        n_particles_y = fluid_particles_per_dimension(fluid_size[2], particle_spacing,
                                                      "fluid height")

        if tank_size[1] < fluid_size[1] - 1e-5 * particle_spacing
            n_particles_x -= 1
            @info "The fluid was overlapping.\n New fluid width is set to $((n_particles_x + 1) * particle_spacing)"
        end

        if tank_size[2] < fluid_size[2] - 1e-5 * particle_spacing
            n_particles_y -= 1
            @info "The fluid was overlapping.\n New fluid height is set to $((n_particles_y + 1) * particle_spacing)"
        end

        n_particles_per_dimension = (n_particles_x, n_particles_y)
        particle_coordinates = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
        particle_velocities = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))

        initialize_particles!(particle_coordinates, particle_velocities, particle_spacing,
                              init_velocity, n_particles_per_dimension)
        particle_densities = fluid_density * ones(Float64,
                                                  prod(n_particles_per_dimension))
        !isempty(pressure) &&
            (pressure = pressure * ones(Float64, prod(n_particles_per_dimension)))

        mass = fluid_density * particle_spacing^2
        particle_masses = mass * ones(ELTYPE, prod(n_particles_per_dimension))

        fluid = InitialCondition(particle_coordinates, particle_velocities,
                                 particle_masses, particle_densities, pressure=pressure)

        boundary = InitialCondition(boundary_coordinates, boundary_velocities,
                                    boundary_masses, boundary_densities, pressure=pressure)

        return new{NDIMS, 2 * NDIMS, ELTYPE}(fluid, boundary,
                                             faces, face_indices,
                                             particle_spacing, spacing_ratio, n_layers,
                                             n_particles_per_dimension)
    end

    function RectangularTank(particle_spacing, fluid_size::NTuple{3}, tank_size,
                             fluid_density; pressure=[],
                             n_layers=1, spacing_ratio=1.0, init_velocity=(0.0, 0.0, 0.0),
                             boundary_density=fluid_density, faces=Tuple(trues(6)))
        NDIMS = 3

        if particle_spacing < eps()
            throw(ArgumentError("Particle spacing needs to be positive and larger than $(eps())!"))
        end

        if fluid_density < eps()
            throw(ArgumentError("Density needs to be positive and larger than $(eps())!"))
        end

        if length(tank_size) != NDIMS
            throw(ArgumentError("`tank_size` must be of length $NDIMS for a $(NDIMS)D problem"))
        end

        ELTYPE = eltype(particle_spacing)

        # Leave space for the fluid particles
        tank_size = tank_size .+ particle_spacing

        # Boundary particle data
        n_boundaries_per_dim,
        tank_size = boundary_particles_per_dimension(tank_size, particle_spacing,
                                                     spacing_ratio)

        boundary_coordinates,
        face_indices = initialize_boundaries(particle_spacing / spacing_ratio, tank_size,
                                             n_boundaries_per_dim, n_layers, faces)

        boundary_masses = boundary_density * (particle_spacing / spacing_ratio)^3 *
                          ones(ELTYPE, size(boundary_coordinates, 2))
        boundary_densities = boundary_density * ones(ELTYPE, size(boundary_coordinates, 2))
        boundary_velocities = zeros(ELTYPE, size(boundary_coordinates))

        # Particle data
        n_particles_x = fluid_particles_per_dimension(fluid_size[1], particle_spacing,
                                                      "fluid width")
        n_particles_y = fluid_particles_per_dimension(fluid_size[2], particle_spacing,
                                                      "fluid height")
        n_particles_z = fluid_particles_per_dimension(fluid_size[3], particle_spacing,
                                                      "fluid depth")

        if tank_size[1] < fluid_size[1] - 1e-5 * particle_spacing
            n_particles_x -= 1
            @info "The fluid was overlapping.\n New fluid width is set to $((n_particles_x + 1) * particle_spacing)"
        end

        if tank_size[2] < fluid_size[2] - 1e-5 * particle_spacing
            n_particles_y -= 1
            @info "The fluid was overlapping.\n New fluid height is set to $((n_particles_y + 1) * particle_spacing)"
        end

        if tank_size[3] < fluid_size[3] - 1e-5 * particle_spacing
            n_particles_z -= 1
            @info "The fluid was overlapping.\n New fluid depth is set to $((n_particles_z + 1) * particle_spacing)"
        end

        n_particles_per_dimension = (n_particles_x, n_particles_y, n_particles_z)

        particle_coordinates = Array{Float64, 2}(undef, 3, prod(n_particles_per_dimension))
        particle_velocities = Array{Float64, 2}(undef, 3, prod(n_particles_per_dimension))

        initialize_particles!(particle_coordinates, particle_velocities, particle_spacing,
                              init_velocity, n_particles_per_dimension)
        particle_densities = fluid_density * ones(Float64, prod(n_particles_per_dimension))
        !isempty(pressure) &&
            (pressure = pressure * ones(Float64, prod(n_particles_per_dimension)))
        mass = fluid_density * particle_spacing^3
        particle_masses = mass * ones(ELTYPE, prod(n_particles_per_dimension))

        fluid = InitialCondition(particle_coordinates, particle_velocities,
                                 particle_masses, particle_densities, pressure=pressure)

        boundary = InitialCondition(boundary_coordinates, boundary_velocities,
                                    boundary_masses, boundary_densities, pressure=pressure)

        return new{NDIMS, 2 * NDIMS, ELTYPE}(fluid, boundary,
                                             faces, face_indices,
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
        particle_velocities[:, particle] .= init_velocity
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
        particle_velocities[:, particle] .= init_velocity
    end
end

# 2D
function initialize_boundaries(particle_spacing, tank_size::NTuple{2},
                               n_boundaries_per_dim, n_layers, faces)
    n_particles_x, n_particles_y = n_boundaries_per_dim

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
        left_boundary = rectangular_shape_coords(particle_spacing,
                                                 (n_layers, n_particles_y),
                                                 (layer_offset, particle_spacing))

        # store coordinates of left boundary
        boundary_coordinates = hcat(boundary_coordinates, left_boundary)

        # store the indices of each particle
        particles_per_layer = n_particles_y
        for i in 1:n_layers
            face_indices_1[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    #### Right boundary
    if faces[2]
        right_boundary = rectangular_shape_coords(particle_spacing,
                                                  (n_layers, n_particles_y),
                                                  (tank_size[1], particle_spacing))

        # store coordinates of left boundary
        boundary_coordinates = hcat(boundary_coordinates, right_boundary)

        # store the indices of each particle
        particles_per_layer = n_particles_y
        for i in 1:n_layers
            face_indices_2[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    #### Bottom boundary
    if faces[3]
        bottom_boundary = rectangular_shape_coords(particle_spacing,
                                                   (n_particles_x, n_layers),
                                                   (particle_spacing, layer_offset),
                                                   loop_order=:y_first)

        # store coordinates of left boundary
        boundary_coordinates = hcat(boundary_coordinates, bottom_boundary)

        # store the indices of each particle
        particles_per_layer = n_particles_x
        for i in 1:n_layers
            face_indices_3[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    #### Top boundary
    if faces[4]
        top_boundary = rectangular_shape_coords(particle_spacing,
                                                (n_particles_x, n_layers),
                                                (particle_spacing, tank_size[2]),
                                                loop_order=:y_first)

        # store coordinates of left boundary
        boundary_coordinates = hcat(boundary_coordinates, top_boundary)

        # store the indices of each particle
        particles_per_layer = n_particles_x
        for i in 1:n_layers
            face_indices_4[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    #### Add corners
    # Bottom left
    if faces[1] && faces[3]
        bottom_left_corner = rectangular_shape_coords(particle_spacing,
                                                      (n_layers, n_layers),
                                                      (layer_offset, layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, bottom_left_corner)
    end

    # Top left
    if faces[1] && faces[4]
        top_left_corner = rectangular_shape_coords(particle_spacing,
                                                   (n_layers, n_layers),
                                                   (layer_offset, tank_size[2]))
        boundary_coordinates = hcat(boundary_coordinates, top_left_corner)
    end

    # Bottom right
    if faces[2] && faces[3]
        bottom_right_corner = rectangular_shape_coords(particle_spacing,
                                                       (n_layers, n_layers),
                                                       (tank_size[1], layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, bottom_right_corner)
    end

    # Top right
    if faces[2] && faces[4]
        top_right_corner = rectangular_shape_coords(particle_spacing,
                                                    (n_layers, n_layers),
                                                    (tank_size[1], tank_size[2]))
        boundary_coordinates = hcat(boundary_coordinates, top_right_corner)
    end

    return boundary_coordinates,
           (face_indices_1, face_indices_2, face_indices_3, face_indices_4)
end

# 3D
function initialize_boundaries(particle_spacing, tank_size::NTuple{3},
                               n_boundaries_per_dim, n_layers, faces)
    n_particles_x, n_particles_y, n_particles_z = n_boundaries_per_dim

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
        x_neg_boundary = rectangular_shape_coords(particle_spacing,
                                                  (n_layers, n_particles_y, n_particles_z),
                                                  (layer_offset, particle_spacing,
                                                   particle_spacing))

        # store coordinates of left boundary
        boundary_coordinates = hcat(boundary_coordinates, x_neg_boundary)

        # store the indices of each particle
        particles_per_layer = prod((n_particles_y, n_particles_z))
        for i in 1:n_layers
            face_indices_1[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    #### +x boundary (y-z-plane)
    if faces[2]
        x_pos_boundary = rectangular_shape_coords(particle_spacing,
                                                  (n_layers, n_particles_y, n_particles_z),
                                                  (tank_size[1], particle_spacing,
                                                   particle_spacing))

        # store coordinates of left boundary
        boundary_coordinates = hcat(boundary_coordinates, x_pos_boundary)

        # store the indices of each particle
        particles_per_layer = prod((n_particles_y, n_particles_z))
        for i in 1:n_layers
            face_indices_2[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    #### -y boundary (x-z-plane)
    if faces[3]
        y_neg_boundary = rectangular_shape_coords(particle_spacing,
                                                  (n_particles_x, n_layers, n_particles_z),
                                                  (particle_spacing, layer_offset,
                                                   particle_spacing),
                                                  loop_order=:y_first)

        # store coordinates of left boundary
        boundary_coordinates = hcat(boundary_coordinates, y_neg_boundary)

        # store the indices of each particle
        particles_per_layer = prod((n_particles_x, n_particles_z))
        for i in 1:n_layers
            face_indices_3[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    #### +y boundary (x-z-plane)
    if faces[4]
        y_pos_boundary = rectangular_shape_coords(particle_spacing,
                                                  (n_particles_x, n_layers, n_particles_z),
                                                  (particle_spacing, tank_size[2],
                                                   particle_spacing), loop_order=:y_first)

        # store coordinates of left boundary
        boundary_coordinates = hcat(boundary_coordinates, y_pos_boundary)

        # store the indices of each particle
        particles_per_layer = prod((n_particles_x, n_particles_z))
        for i in 1:n_layers
            face_indices_4[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    #### -z boundary (x-y-plane).
    if faces[5]
        z_neg_boundary = rectangular_shape_coords(particle_spacing,
                                                  (n_particles_x, n_particles_y, n_layers),
                                                  (particle_spacing, particle_spacing,
                                                   layer_offset), loop_order=:z_first)

        # store coordinates of left boundary
        boundary_coordinates = hcat(boundary_coordinates, z_neg_boundary)

        # store the indices of each particle
        particles_per_layer = prod((n_particles_x, n_particles_y))
        for i in 1:n_layers
            face_indices_5[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    #### +z boundary (x-y-plane)
    if faces[6]
        z_pos_boundary = rectangular_shape_coords(particle_spacing,
                                                  (n_particles_x, n_particles_y, n_layers),
                                                  (particle_spacing, particle_spacing,
                                                   tank_size[3]), loop_order=:z_first)

        # store coordinates of left boundary
        boundary_coordinates = hcat(boundary_coordinates, z_pos_boundary)

        # store the indices of each particle
        particles_per_layer = prod((n_particles_x, n_particles_y))
        for i in 1:n_layers
            face_indices_6[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    #### Add edges
    if faces[1] && faces[3]
        edge_1_3 = rectangular_shape_coords(particle_spacing,
                                            (n_layers, n_layers, n_particles_z),
                                            (layer_offset, layer_offset, particle_spacing))
        boundary_coordinates = hcat(boundary_coordinates, edge_1_3)
    end

    if faces[1] && faces[4]
        edge_1_4 = rectangular_shape_coords(particle_spacing,
                                            (n_layers, n_layers, n_particles_z),
                                            (layer_offset, tank_size[2], particle_spacing))
        boundary_coordinates = hcat(boundary_coordinates, edge_1_4)
    end

    if faces[2] && faces[3]
        edge_2_3 = rectangular_shape_coords(particle_spacing,
                                            (n_layers, n_layers, n_particles_z),
                                            (tank_size[1], layer_offset, particle_spacing))
        boundary_coordinates = hcat(boundary_coordinates, edge_2_3)
    end

    if faces[2] && faces[4]
        edge_2_4 = rectangular_shape_coords(particle_spacing,
                                            (n_layers, n_layers, n_particles_z),
                                            (tank_size[1], tank_size[2], particle_spacing))
        boundary_coordinates = hcat(boundary_coordinates, edge_2_4)
    end

    if faces[5] && faces[3]
        edge_5_3 = rectangular_shape_coords(particle_spacing,
                                            (n_particles_x, n_layers, n_layers),
                                            (particle_spacing, layer_offset, layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, edge_5_3)
    end

    if faces[5] && faces[4]
        edge_5_4 = rectangular_shape_coords(particle_spacing,
                                            (n_particles_x, n_layers, n_layers),
                                            (particle_spacing, tank_size[2], layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, edge_5_4)
    end

    if faces[6] && faces[3]
        edge_6_3 = rectangular_shape_coords(particle_spacing,
                                            (n_particles_x, n_layers, n_layers),
                                            (particle_spacing, layer_offset, tank_size[3]))
        boundary_coordinates = hcat(boundary_coordinates, edge_6_3)
    end

    if faces[6] && faces[4]
        edge_6_4 = rectangular_shape_coords(particle_spacing,
                                            (n_particles_x, n_layers, n_layers),
                                            (particle_spacing, tank_size[2], tank_size[3]))
        boundary_coordinates = hcat(boundary_coordinates, edge_6_4)
    end

    if faces[1] && faces[5]
        edge_1_5 = rectangular_shape_coords(particle_spacing,
                                            (n_layers, n_particles_y, n_layers),
                                            (layer_offset, particle_spacing, layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, edge_1_5)
    end

    if faces[1] && faces[6]
        edge_1_6 = rectangular_shape_coords(particle_spacing,
                                            (n_layers, n_particles_y, n_layers),
                                            (layer_offset, particle_spacing, tank_size[3]))
        boundary_coordinates = hcat(boundary_coordinates, edge_1_6)
    end

    if faces[5] && faces[2]
        edge_5_2 = rectangular_shape_coords(particle_spacing,
                                            (n_layers, n_particles_y, n_layers),
                                            (tank_size[1], particle_spacing, layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, edge_5_2)
    end

    if faces[6] && faces[2]
        edge_6_2 = rectangular_shape_coords(particle_spacing,
                                            (n_layers, n_particles_y, n_layers),
                                            (tank_size[1], particle_spacing, tank_size[3]))
        boundary_coordinates = hcat(boundary_coordinates, edge_6_2)
    end

    #### Add corners
    if faces[1] && faces[3] && faces[5]
        corner_1_3_5 = rectangular_shape_coords(particle_spacing,
                                                (n_layers, n_layers, n_layers),
                                                (layer_offset, layer_offset, layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, corner_1_3_5)
    end

    if faces[1] && faces[4] && faces[5]
        corner_1_4_5 = rectangular_shape_coords(particle_spacing,
                                                (n_layers, n_layers, n_layers),
                                                (layer_offset, tank_size[2], layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, corner_1_4_5)
    end

    if faces[1] && faces[3] && faces[6]
        corner_1_3_6 = rectangular_shape_coords(particle_spacing,
                                                (n_layers, n_layers, n_layers),
                                                (layer_offset, layer_offset, tank_size[3]))
        boundary_coordinates = hcat(boundary_coordinates, corner_1_3_6)
    end

    if faces[1] && faces[4] && faces[6]
        corner_1_4_6 = rectangular_shape_coords(particle_spacing,
                                                (n_layers, n_layers, n_layers),
                                                (layer_offset, tank_size[2], tank_size[3]))
        boundary_coordinates = hcat(boundary_coordinates, corner_1_4_6)
    end

    if faces[2] && faces[3] && faces[5]
        corner_2_3_5 = rectangular_shape_coords(particle_spacing,
                                                (n_layers, n_layers, n_layers),
                                                (tank_size[1], layer_offset, layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, corner_2_3_5)
    end

    if faces[2] && faces[4] && faces[5]
        corner_2_4_5 = rectangular_shape_coords(particle_spacing,
                                                (n_layers, n_layers, n_layers),
                                                (tank_size[1], tank_size[2], layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, corner_2_4_5)
    end

    if faces[2] && faces[3] && faces[6]
        corner_2_3_6 = rectangular_shape_coords(particle_spacing,
                                                (n_layers, n_layers, n_layers),
                                                (tank_size[1], layer_offset, tank_size[3]))
        boundary_coordinates = hcat(boundary_coordinates, corner_2_3_6)
    end

    if faces[2] && faces[4] && faces[6]
        corner_2_4_6 = rectangular_shape_coords(particle_spacing,
                                                (n_layers, n_layers, n_layers),
                                                (tank_size[1], tank_size[2],
                                                 tank_size[3]))
        boundary_coordinates = hcat(boundary_coordinates, corner_2_4_6)
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
    @unpack boundary, particle_spacing, spacing_ratio,
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
                boundary.coordinates[dim, particle] = positions[face] + layer_shift
            end
        end
    end

    return rectangular_tank
end

function fluid_particles_per_dimension(size, spacing, dimension)
    n_particles = round(Int, size / spacing)

    new_size = n_particles * spacing
    if round(new_size, digits=4) != round(size, digits=4)
        print_warn_message(dimension, size, new_size)
    end

    return n_particles
end

function boundary_particles_per_dimension(tank_size::NTuple{2}, particle_spacing,
                                          spacing_ratio)
    n_boundaries_x = round(Int, (tank_size[1] / particle_spacing * spacing_ratio))
    n_boundaries_y = round(Int, (tank_size[2] / particle_spacing * spacing_ratio))

    new_container_width = n_boundaries_x * (particle_spacing / spacing_ratio)
    new_container_height = n_boundaries_y * (particle_spacing / spacing_ratio)

    if round(new_container_width, digits=4) != round(tank_size[1], digits=4)
        print_warn_message("container width", tank_size[1], new_container_width)
    end
    if round(new_container_height, digits=4) != round(tank_size[2], digits=4)
        print_warn_message("container height", tank_size[2], new_container_height)
    end

    # The container size is larger than the fluid area by one particle spacing,
    # since the boundary particles enclose the fluid particles.
    # For the boundary faces we need the size of the fluid area.
    # Thus, remove one particle again.
    n_boundaries_x -= 1
    n_boundaries_y -= 1

    return (n_boundaries_x, n_boundaries_y), (new_container_width, new_container_height)
end

function boundary_particles_per_dimension(tank_size::NTuple{3}, particle_spacing,
                                          spacing_ratio)
    n_boundaries_x = round(Int, tank_size[1] / particle_spacing * spacing_ratio)
    n_boundaries_y = round(Int, tank_size[2] / particle_spacing * spacing_ratio)
    n_boundaries_z = round(Int, tank_size[3] / particle_spacing * spacing_ratio)

    new_container_width = n_boundaries_x * (particle_spacing / spacing_ratio)
    new_container_height = n_boundaries_y * (particle_spacing / spacing_ratio)
    new_container_depth = n_boundaries_z * (particle_spacing / spacing_ratio)

    if round(new_container_width, digits=4) != round(tank_size[1], digits=4)
        print_warn_message("container width", tank_size[1], new_container_width)
    end
    if round(new_container_height, digits=4) != round(tank_size[2], digits=4)
        print_warn_message("container height", tank_size[2], new_container_height)
    end
    if round(new_container_depth, digits=4) != round(tank_size[3], digits=4)
        print_warn_message("container depth", tank_size[3], new_container_depth)
    end

    # The container size is larger than the fluid area by one particle spacing,
    # since the boundary particles enclose the fluid particles.
    # For the boundary faces we need the size of the fluid area.
    # Thus, remove one particle again.
    n_boundaries_x -= 1
    n_boundaries_y -= 1
    n_boundaries_z -= 1

    return (n_boundaries_x, n_boundaries_y, n_boundaries_z),
           (new_container_width, new_container_height, new_container_depth)
end

function print_warn_message(dimension, size, new_size)
    @info "The desired $dimension $size is not a multiple of the particle spacing.\n New $dimension is set to $new_size."
end

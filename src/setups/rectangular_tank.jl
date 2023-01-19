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

See also: [`reset_right_wall!`](@ref)
"""
struct RectangularTank{NDIMS, NDIMSt2, ELTYPE <: Real}
    particle_coordinates       :: Array{ELTYPE, 2}
    particle_velocities        :: Array{ELTYPE, 2}
    particle_densities         :: Vector{ELTYPE}
    particle_masses            :: Vector{ELTYPE}
    boundary_coordinates       :: Array{ELTYPE, 2}
    boundary_masses            :: Vector{ELTYPE}
    faces_                     :: NTuple{NDIMSt2, Bool} # store if face in dir exists (-x +x -y +y -z +z)
    face_indices               :: NTuple{NDIMSt2, Vector{Int}}
    particle_spacing           :: ELTYPE
    spacing_ratio              :: ELTYPE
    n_layers                   :: Int
    n_particles_per_dimension  :: NTuple{NDIMS, Int}
    n_boundaries_per_dimension :: NTuple{NDIMS, Int}

    function RectangularTank(particle_spacing, spacing_ratio, fluid_width, fluid_height,
                             container_width, container_height, fluid_density;
                             n_layers=1, init_velocity=0.0, boundary_density=fluid_density,
                             faces=Tuple(trues(4)))
        NDIMS = 2
        ELTYPE = eltype(particle_spacing)

        # Boundary particle data
        n_boundaries_x, n_boundaries_y, container_width,
        container_height = get_boundary_particles_per_dimension(container_width,
                                                                container_height,
                                                                particle_spacing,
                                                                spacing_ratio)

        boundary_coordinates, face_indices = initialize_boundaries(particle_spacing /
                                                                   spacing_ratio,
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
        n_boundaries_per_dimension = (n_boundaries_x, n_boundaries_y)
        particle_coordinates = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
        particle_velocities = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))

        initialize_particles!(particle_coordinates, particle_velocities, particle_spacing,
                              init_velocity, n_particles_per_dimension)
        particle_densities = fluid_density * ones(Float64, prod(n_particles_per_dimension))
        mass = fluid_density * particle_spacing^2
        particle_masses = mass * ones(ELTYPE, prod(n_particles_per_dimension))

        return new{NDIMS, 2 * NDIMS, ELTYPE}(particle_coordinates, particle_velocities,
                                             particle_densities, particle_masses,
                                             boundary_coordinates, boundary_masses, faces,
                                             face_indices,
                                             particle_spacing, spacing_ratio, n_layers,
                                             n_particles_per_dimension,
                                             n_boundaries_per_dimension)
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

        # Boundary particle data
        n_boundaries_x, n_boundaries_y, n_boundaries_z, container_width, container_height,
        container_depth = get_boundary_particles_per_dimension(container_width,
                                                               container_height,
                                                               container_depth,
                                                               particle_spacing,
                                                               spacing_ratio)

        boundary_coordinates = initialize_boundaries(particle_spacing / spacing_ratio,
                                                     container_width, container_height,
                                                     container_depth,
                                                     n_boundaries_x, n_boundaries_y,
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
        n_boundaries_per_dimension = (n_boundaries_x, n_boundaries_y, n_boundaries_z)

        particle_coordinates = Array{Float64, 2}(undef, 3, prod(n_particles_per_dimension))
        particle_velocities = Array{Float64, 2}(undef, 3, prod(n_particles_per_dimension))

        initialize_particles!(particle_coordinates, particle_velocities, particle_spacing,
                              init_velocity, n_particles_per_dimension)
        particle_densities = fluid_density * ones(Float64, prod(n_particles_per_dimension))
        particle_masses = mass * ones(ELTYPE, prod(n_particles_per_dimension))

        return new{NDIMS, 2 * NDIMS, ELTYPE}(particle_coordinates, particle_velocities,
                                             particle_densities, particle_masses,
                                             boundary_coordinates, boundary_masses, faces,
                                             particle_spacing, spacing_ratio, n_layers,
                                             n_particles_per_dimension,
                                             n_boundaries_per_dimension)
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

function initialize_boundaries(particle_spacing,
                               container_width, container_height,
                               n_particles_x, n_particles_y, n_layers, faces)
    # +-x faces: add cornes in each direction if needed.
    # If -y face exists, add layers in -y-direction. If +y face exists, add layers in +y-direction.
    n_particles_y += (faces[3] + faces[4]) * (n_layers - 1)

    # +-y faces: All corners have already been added above.
    # If +-x face exists, remove one overlapping edge particle in +-x direction.
    n_particles_x -= (faces[1] + faces[2])

    n_particles = n_layers *
                  ((faces[1] + faces[2]) * n_particles_y +
                   (faces[3] + faces[4]) * n_particles_x)

    boundary_coordinates = Array{Float64, 2}(undef, 2, n_particles)

    # Store each index
    face1 = Vector{Int}(undef, 0)
    face2 = Vector{Int}(undef, 0)
    face3 = Vector{Int}(undef, 0)
    face4 = Vector{Int}(undef, 0)

    boundary_particle = 0
    for i in 0:(n_layers - 1)
        # Left boundary
        faces[1] && for y in 1:n_particles_y
            boundary_particle += 1

            boundary_coordinates[1, boundary_particle] = -i * particle_spacing
            # Offset explained above
            boundary_coordinates[2, boundary_particle] = (y - 1 - faces[3] * (n_layers - 1)) *
                                                         particle_spacing

            append!(face1, boundary_particle)
        end

        # Right boundary
        faces[2] && for y in 1:n_particles_y
            boundary_particle += 1

            boundary_coordinates[1, boundary_particle] = container_width +
                                                         i * particle_spacing
            boundary_coordinates[2, boundary_particle] = (y - 1 - faces[3] * (n_layers - 1)) *
                                                         particle_spacing

            append!(face2, boundary_particle)
        end

        # Bottom boundary
        faces[3] && for x in 1:n_particles_x
            boundary_particle += 1

            # Offset explained above
            boundary_coordinates[1, boundary_particle] = (x - 1 + faces[1]) *
                                                         particle_spacing
            boundary_coordinates[2, boundary_particle] = -i * particle_spacing

            append!(face3, boundary_particle)
        end

        # Top boundary
        faces[4] && for x in 1:n_particles_x
            boundary_particle += 1

            boundary_coordinates[1, boundary_particle] = (x - 1 + faces[1]) *
                                                         particle_spacing
            boundary_coordinates[2, boundary_particle] = container_height +
                                                         i * particle_spacing

            append!(face4, boundary_particle)
        end
    end

    return boundary_coordinates, (face1, face2, face3, face4)
end

function initialize_boundaries(particle_spacing,
                               container_width, container_height, container_depth,
                               n_particles_x, n_particles_y, n_particles_z,
                               n_layers, faces)
    # y-z-plane: add edges in each direction if needed
    # If -y face exists, add layers in -y-direction. If +y face exists, add layers in +y-direction.
    # If -z face exists, add layers in -z-direction. If +z face exists, add layers in +z-direction.
    n_particles_y_z = (n_particles_y + (faces[3] + faces[4]) * (n_layers - 1)) *
                      (n_particles_z + (faces[5] + faces[6]) * (n_layers - 1))

    # x-z-plane: add edges in +-z direction if needed. Edges in +-x direction have been added above.
    # If +-x face exists, remove one overlapping edge particle in +-x direction.
    n_particles_x_z = (n_particles_x - (faces[1] + faces[2])) *
                      (n_particles_z + (faces[5] + faces[6]) * (n_layers - 1))

    # x-y-plane: All edges have already been added above.
    # If +-x face exists, remove one overlapping edge particle in +-x direction.
    # If +-y face exists, remove one overlapping edge particle in +-y direction.
    n_particles_x_y = (n_particles_x - (faces[1] + faces[2])) *
                      (n_particles_y - (faces[3] + faces[4]))

    n_particles = n_layers *
                  ((faces[1] + faces[2]) * n_particles_y_z +
                   (faces[3] + faces[4]) * n_particles_x_z +
                   (faces[5] + faces[6]) * n_particles_x_y)

    boundary_coordinates = Array{Float64, 2}(undef, 3, n_particles)

    # Store each index
    face1 = Vector{Int}(undef, 0)
    face2 = Vector{Int}(undef, 0)
    face3 = Vector{Int}(undef, 0)
    face4 = Vector{Int}(undef, 0)
    face5 = Vector{Int}(undef, 0)
    face6 = Vector{Int}(undef, 0)

    boundary_particle = 0
    for i in 0:(n_layers - 1)
        # -x boundary (y-z-plane). See explanation above.
        n_particles_z_ = n_particles_z + (faces[5] + faces[6]) * (n_layers - 1)
        n_particles_y_ = n_particles_y + (faces[3] + faces[4]) * (n_layers - 1)
        faces[1] && for z in 1:n_particles_z_, y in 1:n_particles_y_
            boundary_particle += 1

            boundary_coordinates[1, boundary_particle] = -i * particle_spacing
            boundary_coordinates[2, boundary_particle] = (y - 1 - faces[3] * (n_layers - 1)) *
                                                         particle_spacing
            boundary_coordinates[3, boundary_particle] = (z - 1 - faces[5] * (n_layers - 1)) *
                                                         particle_spacing

            append!(face1, boundary_particle)
        end

        # +x boundary (y-z-plane)
        faces[2] && for z in 1:n_particles_z_, y in 1:n_particles_y_
            boundary_particle += 1

            boundary_coordinates[1, boundary_particle] = container_width +
                                                         i * particle_spacing
            boundary_coordinates[2, boundary_particle] = (y - 1 - faces[3] * (n_layers - 1)) *
                                                         particle_spacing
            boundary_coordinates[3, boundary_particle] = (z - 1 - faces[5] * (n_layers - 1)) *
                                                         particle_spacing

            append!(face2, boundary_particle)
        end

        # -y boundary (x-z-plane). See explanation above.
        n_particles_z_ = n_particles_z + (faces[5] + faces[6]) * (n_layers - 1)
        n_particles_x_ = n_particles_x - (faces[1] + faces[2])
        faces[3] && for z in 1:n_particles_z_, x in 1:n_particles_x_
            boundary_particle += 1

            boundary_coordinates[1, boundary_particle] = (x - 1 + faces[1]) *
                                                         particle_spacing
            boundary_coordinates[2, boundary_particle] = -i * particle_spacing
            boundary_coordinates[3, boundary_particle] = (z - 1 - faces[5] * (n_layers - 1)) *
                                                         particle_spacing

            append!(face3, boundary_particle)
        end

        # +y boundary (x-z-plane)
        faces[4] && for z in 1:n_particles_z_, x in 1:n_particles_x_
            boundary_particle += 1

            boundary_coordinates[1, boundary_particle] = (x - 1 + faces[1]) *
                                                         particle_spacing
            boundary_coordinates[2, boundary_particle] = container_height +
                                                         i * particle_spacing
            boundary_coordinates[3, boundary_particle] = (z - 1 - faces[5] * (n_layers - 1)) *
                                                         particle_spacing

            append!(face4, boundary_particle)
        end

        # -z boundary (x-y-plane). See explanation above.
        n_particles_y_ = n_particles_y - (faces[3] + faces[4])
        n_particles_x_ = n_particles_x - (faces[1] + faces[2])
        faces[5] && for y in 1:n_particles_y_, x in 1:n_particles_x_
            boundary_particle += 1

            boundary_coordinates[1, boundary_particle] = (x - 1 + faces[1]) *
                                                         particle_spacing
            boundary_coordinates[2, boundary_particle] = (y - 1 + faces[3]) *
                                                         particle_spacing
            boundary_coordinates[3, boundary_particle] = -i * particle_spacing

            append!(face5, boundary_particle)
        end

        # +z boundary (x-y-plane)
        faces[6] && for y in 1:n_particles_y_, x in 1:n_particles_x_
            boundary_particle += 1

            boundary_coordinates[1, boundary_particle] = (x - 1 + faces[1]) *
                                                         particle_spacing
            boundary_coordinates[2, boundary_particle] = (y - 1 + faces[3]) *
                                                         particle_spacing
            boundary_coordinates[3, boundary_particle] = container_depth +
                                                         i * particle_spacing

            append!(face6, boundary_particle)
        end
    end

    return boundary_coordinates, (face1, face2, face3, face4, face5, face6)
end

@doc raw"""
    reset_right_wall!(rectangular_tank::RectangularTank, container_width;
                      wall_position=container_width, n_layers=1)

The right wall of the tank will be set to a desired position by calling the function with the keyword argument `wall_position`, which
is the ``x`` coordinate of the desired position.
"""
function reset_wall!(rectangular_tank, reset_face, position)
    @unpack boundary_coordinates, particle_spacing, spacing_ratio,
    n_layers, face_indices = rectangular_tank

    dim = 1
    for j in eachindex(reset_face)
        sizes = [round(Int, length(face_indices[j]) / n_layers) for i in 1:n_layers]
        ranges = Tuple((sum(sizes[1:(i - 1)]) + 1):sum(sizes[1:i])
                       for i in eachindex(sizes))

        reset_face[j] && for i in 0:(n_layers - 1)
            for bound_index in face_indices[j][ranges[i + 1]]
                boundary_coordinates[dim, bound_index] = position[j] +
                                                         i * particle_spacing /
                                                         spacing_ratio
            end
        end
        dim += iseven(j)
    end
end

function get_fluid_particles_per_dimension(size, spacing, dimension)
    # remove one particle, otherwise the fluid particles are placed in the boundary region
    n_particles = round(Int, size / spacing) - 1

    new_size = (n_particles + 1) * spacing
    if round(new_size, digits=4) != round(size, digits=4)
        print_warn_message(dimension, size, new_size)
    end

    return n_particles
end

function get_boundary_particles_per_dimension(container_width, container_height,
                                              particle_spacing, spacing_ratio)
    n_boundaries_x = round(Int, (container_width / particle_spacing * spacing_ratio)) + 1
    n_boundaries_y = round(Int, (container_height / particle_spacing * spacing_ratio)) + 1

    new_container_width = (n_boundaries_x - 1) * (particle_spacing / spacing_ratio)
    new_container_height = (n_boundaries_y - 1) * (particle_spacing / spacing_ratio)

    if round(new_container_width, digits=4) != round(container_width, digits=4)
        print_warn_message("container width", container_width, new_container_width)
    end
    if round(new_container_height, digits=4) != round(container_height, digits=4)
        print_warn_message("container height", container_height, new_container_height)
    end

    return n_boundaries_x, n_boundaries_y, new_container_width, new_container_height
end

function get_boundary_particles_per_dimension(container_width, container_height,
                                              container_depth,
                                              particle_spacing, spacing_ratio)
    n_boundaries_x = round(Int, container_width / particle_spacing * spacing_ratio) + 1
    n_boundaries_y = round(Int, container_height / particle_spacing * spacing_ratio) + 1
    n_boundaries_z = round(Int, container_depth / particle_spacing * spacing_ratio) + 1

    new_container_width = (n_boundaries_x - 1) * (particle_spacing / spacing_ratio)
    new_container_height = (n_boundaries_y - 1) * (particle_spacing / spacing_ratio)
    new_container_depth = (n_boundaries_z - 1) * (particle_spacing / spacing_ratio)

    if round(new_container_width, digits=4) != round(container_width, digits=4)
        print_warn_message("container width", container_width, new_container_width)
    end
    if round(new_container_height, digits=4) != round(container_height, digits=4)
        print_warn_message("container height", container_height, new_container_height)
    end
    if round(new_container_depth, digits=4) != round(container_depth, digits=4)
        print_warn_message("container depth", container_depth, new_container_depth)
    end

    return n_boundaries_x, n_boundaries_y, n_boundaries_z,
           new_container_width, new_container_height, new_container_depth
end

function print_warn_message(dimension, size, new_size)
    @info "The desired $dimension $size is not a multiple of the particle spacing.\n New $dimension is set to $new_size."
end

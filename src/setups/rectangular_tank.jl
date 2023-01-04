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
The arguments are as follows:
- `particle_spacing`:                    Spacing betweeen the fluid particles
- `spacing_ratio`:                       Ratio of `particle_spacing` to boundary particle spacing.
- `fluid_width`, `fluid_height`:         Initial width and height of the fluid system, respectively.
- `container_width`, `container_height`: Initial width and height of the container, respectively.
- `fluid_density`:                       The rest density of the fluid.
A 3D tank is generated by calling the function additonally with the container and fluid depth (see examples below).

The keyword arguments are as follows:
- `n_layers`:           Number of boundary layers.
- `init_velocity`:      Initial velocity of the fluid particles.
- `boundary_density`:   Density of the boundary particles (by default set to the rest density)

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
struct RectangularTank{NDIMS, ELTYPE<:Real}
    particle_coordinates    ::Array{ELTYPE, 2}
    particle_velocities     ::Array{ELTYPE, 2}
    particle_densities      ::Vector{ELTYPE}
    particle_masses         ::Vector{ELTYPE}
    boundary_coordinates    ::Array{ELTYPE, 2}
    boundary_masses         ::Vector{ELTYPE}
    faces_                  ::Array{Bool, 1} # store if face in dir exists (-x +x -y +y -z +z)
    particle_spacing        ::ELTYPE
    spacing_ratio           ::ELTYPE
    n_layers                ::Int
    n_boundaries_x          ::Int
    n_boundaries_y          ::Int
    n_boundaries_z          ::Int

    function RectangularTank(particle_spacing, spacing_ratio, fluid_width, fluid_height,
                             container_width, container_height, fluid_density;
                             n_layers=1, init_velocity=0.0, boundary_density=fluid_density, faces=trues(4))
        NDIMS = 2
        ELTYPE = eltype(particle_spacing)

        # Boundary particle data
        n_boundaries_x,
            n_boundaries_y = get_boundary_particles_per_dimension(container_width, container_height,
                                                                  particle_spacing, spacing_ratio, n_layers)
        n_boundaries = (((faces[1] + faces[2]) * n_boundaries_y) * n_layers 
                        +((faces[3] + faces[4]) * n_boundaries_x) * n_layers)

        boundary_coordinates = Array{Float64, 2}(undef, 2, n_boundaries)

        initialize_boundaries!(boundary_coordinates, particle_spacing, spacing_ratio,
                               n_boundaries_x, n_boundaries_y, n_layers, faces)
        boundary_masses = boundary_density * (particle_spacing / spacing_ratio)^2 * ones(ELTYPE, n_boundaries)

        # Particle data
        n_particles_x = get_fluid_particles_per_dimension(fluid_width, particle_spacing, "fluid width")
        n_particles_y = get_fluid_particles_per_dimension(fluid_height, particle_spacing, "fluid height")

        if container_width == fluid_width
            n_particles_x = check_overlapping(n_particles_x, n_boundaries_x,
                                              particle_spacing, spacing_ratio, n_layers, "width")
        end

        n_particles_per_dimension = (n_particles_x, n_particles_y)

        particle_coordinates = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
        particle_velocities = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))

        initialize_particles!(particle_coordinates, particle_velocities, particle_spacing,
                              init_velocity, n_particles_per_dimension)
        particle_densities = fluid_density * ones(Float64, prod(n_particles_per_dimension))
        mass = fluid_density * particle_spacing^2
        particle_masses = mass * ones(ELTYPE, prod(n_particles_per_dimension))

        return new{NDIMS, ELTYPE}(particle_coordinates, particle_velocities, particle_densities, particle_masses,
                                  boundary_coordinates, boundary_masses, faces, particle_spacing, spacing_ratio, n_layers,
                                  n_boundaries_x, n_boundaries_y, 0)
    end

    function RectangularTank(particle_spacing, spacing_ratio,
                             fluid_width, fluid_height, fluid_depth,
                             container_width, container_height, container_depth,
                             fluid_density;
                             n_layers=1, init_velocity=0.0, boundary_density=fluid_density, faces=trues(6))
        NDIMS = 3
        ELTYPE = eltype(particle_spacing)
        mass = fluid_density * particle_spacing^3

        # Boundary particle data
        n_boundaries_x, n_boundaries_y,
            n_boundaries_z = get_boundary_particles_per_dimension(container_width, container_height, container_depth,
                                                                  particle_spacing, spacing_ratio, n_layers)
        n_boundaries   = n_layers *
                         ((faces[1] + faces[2]) * (n_boundaries_z-(2*n_layers-1)) * n_boundaries_y     # y-z-plane
                          + (faces[5] + faces[6]) * n_boundaries_x * n_boundaries_y                    # x-y-plane
                          + (faces[3] + faces[4]) * n_boundaries_x * n_boundaries_z)                   # x-z plane

        boundary_coordinates = Array{Float64, 2}(undef, 3, n_boundaries)

        initialize_boundaries!(boundary_coordinates, particle_spacing, spacing_ratio,
                               n_boundaries_x, n_boundaries_y, n_boundaries_z, n_layers, faces)
        boundary_masses = boundary_density * (particle_spacing/spacing_ratio)^3 * ones(ELTYPE, n_boundaries)

        # Particle data
        n_particles_x = get_fluid_particles_per_dimension(fluid_width, particle_spacing, "fluid width")
        n_particles_y = get_fluid_particles_per_dimension(fluid_height, particle_spacing, "fluid heigth")
        n_particles_z = get_fluid_particles_per_dimension(fluid_depth, particle_spacing, "fluid depth")

        if container_width == fluid_width
            n_particles_x = check_overlapping(n_particles_x, n_boundaries_x,
                                              particle_spacing, spacing_ratio, n_layers, "width")
        end

        if container_depth == fluid_depth
            n_particles_z = check_overlapping(n_particles_z, n_boundaries_z,
                                              particle_spacing, spacing_ratio, n_layers, "depth")
        end

        n_particles_per_dimension = (n_particles_x, n_particles_y, n_particles_z)

        particle_coordinates = Array{Float64, 2}(undef, 3, prod(n_particles_per_dimension))
        particle_velocities = Array{Float64, 2}(undef, 3, prod(n_particles_per_dimension))

        initialize_particles!(particle_coordinates, particle_velocities, particle_spacing,
                              init_velocity, n_particles_per_dimension)
        particle_densities = fluid_density * ones(Float64, prod(n_particles_per_dimension))
        particle_masses = mass * ones(ELTYPE, prod(n_particles_per_dimension))

        return new{NDIMS, ELTYPE}(particle_coordinates, particle_velocities, particle_densities, particle_masses,
                                  boundary_coordinates, boundary_masses, faces, particle_spacing, spacing_ratio, n_layers,
                                  n_boundaries_x, n_boundaries_y, n_boundaries_z)
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


function initialize_boundaries!(boundary_coordinates, particle_spacing, spacing_ratio,
                                n_boundaries_x, n_boundaries_y, n_layers, faces)
    boundary_particle_spacing = particle_spacing / spacing_ratio

    boundary_particle = 0
    for i in 0:n_layers-1
        # Left boundary
        if faces[1]
            for y in 1:n_boundaries_y
                boundary_particle += 1

                boundary_coordinates[1, boundary_particle] = 0 - i*boundary_particle_spacing
                boundary_coordinates[2, boundary_particle] = y * boundary_particle_spacing
            end
        end

        # Right boundary
        if faces[2]
            for y in 1:n_boundaries_y
                boundary_particle += 1

                boundary_coordinates[1, boundary_particle] = ((n_boundaries_x-(2*n_layers-1))
                                                            * boundary_particle_spacing
                                                            + i*boundary_particle_spacing)
                boundary_coordinates[2, boundary_particle] = y * boundary_particle_spacing
            end
        end

        # Bottom boundary
        if faces[3]
            for x in 1:n_boundaries_x
                boundary_particle += 1

                boundary_coordinates[1, boundary_particle] = (x * boundary_particle_spacing
                                                            - n_layers*boundary_particle_spacing)
                boundary_coordinates[2, boundary_particle] = -i*boundary_particle_spacing
            end
        end

        # top boundary
        if faces[4]
            for x in 1:n_boundaries_x
                boundary_particle += 1

                boundary_coordinates[1, boundary_particle] = (x * boundary_particle_spacing
                                                                - n_layers*boundary_particle_spacing)
                boundary_coordinates[2, boundary_particle] =((n_boundaries_y-(2*n_layers-1))
                                                            * boundary_particle_spacing
                                                            + i*boundary_particle_spacing)
            end
        end
    end
end

function initialize_boundaries!(boundary_coordinates, particle_spacing, spacing_ratio,
                                n_boundaries_x, n_boundaries_y, n_boundaries_z, n_layers, faces)
    boundary_particle_spacing = particle_spacing/spacing_ratio

    boundary_particle = 0
    for i in 0:n_layers-1
        # -x boundary (y-z-plane)
        if faces[1]
            for z in 1:n_boundaries_z-(2*n_layers-1), y in 1:n_boundaries_y
                boundary_particle += 1

                boundary_coordinates[1, boundary_particle] = 0 - i*boundary_particle_spacing
                boundary_coordinates[2, boundary_particle] = y * boundary_particle_spacing
                boundary_coordinates[3, boundary_particle] = z * boundary_particle_spacing
            end
        end

        # +x boundary (y-z-plane)
        if faces[2]
            for z in 1:n_boundaries_z-(2*n_layers-1), y in 1:n_boundaries_y
                boundary_particle += 1

                boundary_coordinates[1, boundary_particle] = ((n_boundaries_x-(2*n_layers-1))
                                                            * boundary_particle_spacing
                                                            + i * boundary_particle_spacing)
                boundary_coordinates[2, boundary_particle] = y * boundary_particle_spacing
                boundary_coordinates[3, boundary_particle] = z * boundary_particle_spacing
            end
        end

        # - y boundary (x-z-plane)
        if faces[3]
            for z in 1:n_boundaries_z, x in 1:n_boundaries_x
                boundary_particle += 1

                boundary_coordinates[1, boundary_particle] = (x * boundary_particle_spacing
                                                            - n_layers*boundary_particle_spacing)
                boundary_coordinates[2, boundary_particle] = -i * boundary_particle_spacing
                boundary_coordinates[3, boundary_particle] = (z * boundary_particle_spacing
                                                            - n_layers*boundary_particle_spacing)
            end
        end

        # +y boundary (x-z-plane)
        if faces[4]
            for z in 1:n_boundaries_z, x in 1:n_boundaries_x
                boundary_particle += 1

                boundary_coordinates[1, boundary_particle] = (x * boundary_particle_spacing
                                                            - n_layers*boundary_particle_spacing)
                boundary_coordinates[2, boundary_particle] = ((n_boundaries_y-(2*n_layers-1))
                                                            * boundary_particle_spacing
                                                            + i * boundary_particle_spacing)
                boundary_coordinates[3, boundary_particle] = (z * boundary_particle_spacing
                                                            - n_layers*boundary_particle_spacing)
            end
        end

        # -z boundary (x-y-plane)
        if faces[5]
            for y in 1:n_boundaries_y, x in 1:n_boundaries_x
                boundary_particle += 1
    
                boundary_coordinates[1, boundary_particle] = (x * boundary_particle_spacing
                                                                - n_layers*boundary_particle_spacing)
                boundary_coordinates[2, boundary_particle] = y * boundary_particle_spacing
                boundary_coordinates[3, boundary_particle] = 0 - i*boundary_particle_spacing
            end
        end
    
        # +z boundary (x-y-plane)
        if faces[6]
            for y in 1:n_boundaries_y, x in 1:n_boundaries_x
                boundary_particle += 1

                boundary_coordinates[1, boundary_particle] = (x * boundary_particle_spacing
                                                                - n_layers*boundary_particle_spacing)
                boundary_coordinates[2, boundary_particle] = y * boundary_particle_spacing
                boundary_coordinates[3, boundary_particle] = ((n_boundaries_z-(2*n_layers-1))
                                                                * boundary_particle_spacing
                                                                + i*boundary_particle_spacing)
            end
        end
    end
end


@doc raw"""
    reset_right_wall!(rectangular_tank::RectangularTank, container_width;
                      wall_position=container_width, n_layers=1)

The right wall of the tank will be set to a desired position by calling the function with the keyword argument `wall_position`, which
is the ``x`` coordinate of the desired position.
"""
function reset_right_wall!(rectangular_tank::RectangularTank{2}, container_width;
                           wall_position=container_width)
    @unpack boundary_coordinates, particle_spacing, spacing_ratio,
            n_layers, n_boundaries_x, n_boundaries_y = rectangular_tank

    for i in 0:n_layers-1
        for y in 1:n_boundaries_y
            boundary_particle = n_boundaries_y * (1+i) + n_boundaries_x * i + n_boundaries_y * i + y
            boundary_coordinates[1, boundary_particle] = wall_position + i*particle_spacing/spacing_ratio
        end
    end
end

#3D
function reset_right_wall!(rectangular_tank::RectangularTank{3}, container_width;
                           wall_position=container_width)
    @unpack boundary_coordinates, particle_spacing, spacing_ratio,
            n_layers, n_boundaries_x, n_boundaries_y, n_boundaries_z = rectangular_tank

    # +x boundary (y-z-plane)
    for i in 0:n_layers-1
        boundary_particle = (( n_boundaries_z - (2*n_layers-1) ) * n_boundaries_y * (1+i)
                             + ( n_boundaries_z - (2*n_layers-1) ) * n_boundaries_y * i
                             + ( n_boundaries_y * n_boundaries_x * 2 + n_boundaries_z * n_boundaries_x ) * i)
        for z in 1:n_boundaries_z, y in 1:n_boundaries_y
            boundary_particle += 1
            boundary_coordinates[1, boundary_particle] = wall_position + i*particle_spacing/spacing_ratio
        end
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
                                              particle_spacing, spacing_ratio, n_layers)
    n_boundaries_x = round(Int, (container_width / particle_spacing * spacing_ratio)) + 2*n_layers-1
    n_boundaries_y = round(Int, (container_height / particle_spacing * spacing_ratio)) + 2*n_layers-1

    new_container_width = (n_boundaries_x - 2*n_layers+1) * (particle_spacing / spacing_ratio)
    new_container_height = (n_boundaries_y -2*n_layers+1) * (particle_spacing / spacing_ratio)

    if round(new_container_width, digits=4) != round(container_width, digits=4)
        print_warn_message("container width", container_width, new_container_width)
    end
    if round(new_container_height, digits=4) != round(container_height, digits=4)
        print_warn_message("container height", container_height, new_container_height)
    end

    return n_boundaries_x, n_boundaries_y
end

function get_boundary_particles_per_dimension(container_width, container_height, container_depth,
                                              particle_spacing, spacing_ratio, n_layers)
    n_boundaries_x = round(Int, container_width / particle_spacing * spacing_ratio) + 2*n_layers-1
    n_boundaries_y = round(Int, container_height / particle_spacing * spacing_ratio) + 2*n_layers-1
    n_boundaries_z = round(Int, container_depth / particle_spacing * spacing_ratio) + 2*n_layers-1

    new_container_width = (n_boundaries_x - 2*n_layers+1) *  (particle_spacing / spacing_ratio)
    new_container_height = (n_boundaries_y - 2*n_layers+1) *  (particle_spacing / spacing_ratio)
    new_container_depth = (n_boundaries_z - 2*n_layers+1) *  (particle_spacing / spacing_ratio)

    if round(new_container_width, digits=4) != round(container_width, digits=4)
        print_warn_message("container width", container_width, new_container_width)
    end
    if round(new_container_height, digits=4) != round(container_height, digits=4)
        print_warn_message("container height", container_height, new_container_height)
    end
    if round(new_container_depth, digits=4) != round(container_depth, digits=4)
        print_warn_message("container depth", container_depth, new_container_depth)
    end

    return n_boundaries_x, n_boundaries_y, n_boundaries_z
end


function check_overlapping(n_particles, n_boundaries, particle_spacing, spacing_ratio, n_layers, dimension)
    new_container_width = (n_boundaries - 2*n_layers+1) *  (particle_spacing / spacing_ratio)

    if n_particles * particle_spacing > new_container_width - particle_spacing + 1e-5*(particle_spacing / spacing_ratio)
        n_particles -= 1
        @info "The fluid was overlapping.\n New fluid $dimension is set to $((n_particles + 1) * particle_spacing)"
    end

    return n_particles
end


function print_warn_message(dimension, size, new_size)
    @info "The desired $dimension $size is not a multiple of the particle spacing.\n New $dimension is set to $new_size."
end
